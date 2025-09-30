import sys
import os
import base64
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix

# NEW: Roboflow SDK
from inference_sdk import InferenceHTTPClient
from requests import RequestException

# -------- Console Unicode (Windows) --------
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter calls
# Limit upload size (10 MB)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
# Make Flask aware of Railway proxy so scheme/host are correct
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# -------- Roboflow / Workflow Configuration (env-driven) --------
ROBOFLOW_API_KEY   = os.getenv("ROBOFLOW_API_KEY", "REPLACE_ME")
ROBOFLOW_API_BASE  = os.getenv("ROBOFLOW_API_BASE", "https://serverless.roboflow.com")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "masid3")
ROBOFLOW_WORKFLOW  = os.getenv("ROBOFLOW_WORKFLOW", "detect-count-and-visualize")

# Initialize SDK client (will raise if URL is malformed)
rf_client = InferenceHTTPClient(api_url=ROBOFLOW_API_BASE, api_key=ROBOFLOW_API_KEY)

# -------- Storage for processed images --------
# On Railway, /tmp is a safe writable location
PROCESSED_FOLDER = os.getenv("PROCESSED_FOLDER", "/tmp/processed")
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

# -------- File validation --------
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.get("/")
def home():
    return jsonify({"message": "Flask Roboflow API is running!"}), 200

@app.get("/api/health")
def health():
    # quick configuration check
    status = "ok" if ROBOFLOW_API_KEY not in (None, "", "REPLACE_ME") else "missing_roboflow_key"
    return jsonify({
        "status": status,
        "api_base": ROBOFLOW_API_BASE,
        "workspace": ROBOFLOW_WORKSPACE,
        "workflow": ROBOFLOW_WORKFLOW
    }), 200

@app.get("/processed/<path:filename>")
def serve_processed(filename):
    return send_from_directory(app.config["PROCESSED_FOLDER"], filename)

@app.post("/api/detect")
def detect_image():
    # ---- validations
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    if ROBOFLOW_API_KEY in (None, "", "REPLACE_ME"):
        return jsonify({"error": "Server misconfigured: set ROBOFLOW_API_KEY"}), 500

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(image_file.filename):
        return jsonify({"error": "Unsupported file format. Use PNG/JPG."}), 400

    # Optional: return base64 in response? (default true for compatibility)
    include_b64 = (request.args.get("include_base64", "true").lower() != "false")

    # ---- read file to memory
    filename = secure_filename(image_file.filename)
    try:
        file_bytes = image_file.read()
    except Exception as e:
        return jsonify({"error": f"Failed to read image: {e}"}), 500

    # Write a temporary upload file because the SDK expects a path/bytes-like file reference.
    # Using a temp file keeps memory low and is robust across SDK versions.
    temp_upload_path = os.path.join(
        app.config["PROCESSED_FOLDER"],
        f"upload_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{filename}"
    )
    try:
        with open(temp_upload_path, "wb") as f:
            f.write(file_bytes)
    except Exception as e:
        return jsonify({"error": f"Failed to persist upload: {e}"}), 500

    # ---- call Roboflow workflow via SDK
    app.logger.info(
        f"Running workflow via SDK: base={ROBOFLOW_API_BASE}, "
        f"workspace={ROBOFLOW_WORKSPACE}, workflow={ROBOFLOW_WORKFLOW}"
    )
    try:
        # images expects a dict; we pass the local file path
        rf_result = rf_client.run_workflow(
            workspace_name=ROBOFLOW_WORKSPACE,
            workflow_id=ROBOFLOW_WORKFLOW,
            images={"image": temp_upload_path},
            use_cache=True  # cache workflow definition ~15 min
        )
    except Exception as e:
        # SDK wraps network and API errors as generic Exceptions; surface message
        try:
            # Best effort: include class name for quicker triage
            err_name = e.__class__.__name__
            err_msg = f"{err_name}: {e}"
        except Exception:
            err_msg = str(e)
        return jsonify({"error": "Roboflow error (SDK)", "message": err_msg}), 502
    finally:
        # remove temp upload file
        try:
            os.remove(temp_upload_path)
        except Exception:
            pass

    # ---- parse Roboflow response (SDK returns a dict like the REST one)
    try:
        # Expected structure: { "outputs": [ { "output_image": <dict|str>, "predictions": <dict|list> } ] }
        outputs = rf_result.get("outputs", [])
        if not outputs or not isinstance(outputs, list):
            return jsonify({"error": "Unexpected Roboflow response: outputs missing"}), 502

        first_output = outputs[0]

        # Output image may be a dict with 'value' or a raw base64 string
        processed_image_base64 = None
        out_img = first_output.get("output_image")
        if isinstance(out_img, dict):
            processed_image_base64 = out_img.get("value")
        elif isinstance(out_img, str) and out_img.strip():
            processed_image_base64 = out_img.strip()

        # Predictions may be dict { predictions: [...] } or a list directly
        predictions_block = first_output.get("predictions", {})
        if isinstance(predictions_block, dict):
            predictions = predictions_block.get("predictions", []) or []
        elif isinstance(predictions_block, list):
            predictions = predictions_block
        else:
            predictions = []

        # Count per class
        class_counts = {}
        for obj in predictions:
            if isinstance(obj, dict):
                cls = obj.get("class", "Unknown")
                class_counts[cls] = class_counts.get(cls, 0) + 1

        detected_ingredients = sorted(list(class_counts.keys()))
        details = [{"class": c, "count": n} for c, n in class_counts.items()]
        total_ingredients = sum(class_counts.values())

        # ---- save processed image (if provided); else keep original
        processed_filename = f"processed_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{filename}"
        to_write_bytes = None
        if processed_image_base64 and processed_image_base64.strip():
            try:
                to_write_bytes = base64.b64decode(processed_image_base64)
            except Exception:
                to_write_bytes = None  # fall back to original below

        try:
            with open(os.path.join(app.config["PROCESSED_FOLDER"], processed_filename), "wb") as out:
                out.write(to_write_bytes if to_write_bytes is not None else file_bytes)
        except Exception as e:
            return jsonify({"error": f"Failed to write processed image: {e}"}), 500

        # ---- build absolute URL with correct scheme behind Railway proxy
        scheme = request.headers.get("X-Forwarded-Proto", request.scheme)
        processed_image_url = f"{scheme}://{request.host}/processed/{processed_filename}"

        # ---- compatibility 'result' field (string for single class, dict otherwise)
        if len(class_counts) == 1:
            only = next(iter(class_counts))
            result = f"{class_counts[only]} {only}"
        else:
            result = {"ingredients": total_ingredients, "details": details}

        resp_payload = {
            "processed_image_url": processed_image_url,
            "detected_ingredients": detected_ingredients,
            "ingredients": total_ingredients,
            "details": details,
            "result": result,
        }
        if include_b64:
            resp_payload["processed_image_base64"] = processed_image_base64

        return jsonify(resp_payload), 200

    except Exception as e:
        return jsonify({"error": f"Error processing Roboflow response: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
