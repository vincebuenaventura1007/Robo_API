import sys
import os
import base64
import requests
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix

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

# -------- Roboflow API Configuration --------
ROBOFLOW_API_URL = os.getenv(
    "ROBOFLOW_API_URL",
    "https://detect.roboflow.com/infer/workflows/masid3/detect-count-and-visualize"
)
# IMPORTANT: set this in Railway → Variables; default is a placeholder
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "REPLACE_ME")

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
    return jsonify({"status": status}), 200

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
        base64_image = base64.b64encode(file_bytes).decode("utf-8")
    except Exception as e:
        return jsonify({"error": f"Failed to read image: {e}"}), 500

    # ---- call Roboflow workflow
    payload = {
        "api_key": ROBOFLOW_API_KEY,
        "inputs": {
            "image": {"type": "base64", "value": base64_image}
        },
    }

    try:
        rf_resp = requests.post(
            ROBOFLOW_API_URL,
            json=payload,
            timeout=(10, 60),  # 10s connect, 60s read
        )
    except requests.RequestException as e:
        return jsonify({"error": f"Roboflow request failed: {e}"}), 502

    if rf_resp.status_code != 200:
        # Return Roboflow message to help debugging
        try:
            msg = rf_resp.json()
        except Exception:
            msg = rf_resp.text[:500]
        return jsonify({"error": "Roboflow error", "status": rf_resp.status_code, "message": msg}), 502

    # ---- parse Roboflow response
    try:
        data = rf_resp.json()  # expect { outputs: [ { output_image: {...}, predictions: {...} } ] }
        outputs = data.get("outputs", [])
        if not outputs or not isinstance(outputs, list):
            return jsonify({"error": "Unexpected Roboflow response: outputs missing"}), 502

        first_output = outputs[0]

        # Output image (base64) from workflow
        processed_image_base64 = None
        out_img = first_output.get("output_image")
        if isinstance(out_img, dict):
            processed_image_base64 = out_img.get("value")

        # Predictions may be nested: predictions: { predictions: [...] }
        predictions_block = first_output.get("predictions", {})
        predictions = []
        if isinstance(predictions_block, dict):
            predictions = predictions_block.get("predictions", [])
        elif isinstance(predictions_block, list):
            # in case the workflow returns a list directly
            predictions = predictions_block

        # Count per class + uniq list for convenience
        class_counts = {}
        for obj in predictions or []:
            if isinstance(obj, dict):
                cls = obj.get("class", "Unknown")
                class_counts[cls] = class_counts.get(cls, 0) + 1

        detected_ingredients = sorted(list(class_counts.keys()))
        details = [{"class": c, "count": n} for c, n in class_counts.items()]
        total_ingredients = sum(class_counts.values())

        # ---- save processed image (if provided)
        processed_filename = f"processed_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{filename}"
        if processed_image_base64:
            try:
                with open(os.path.join(app.config["PROCESSED_FOLDER"], processed_filename), "wb") as out:
                    out.write(base64.b64decode(processed_image_base64))
            except Exception as e:
                return jsonify({"error": f"Failed to write processed image: {e}"}), 500
        else:
            # fallback: store the original if workflow didn't return an image
            try:
                with open(os.path.join(app.config["PROCESSED_FOLDER"], processed_filename), "wb") as out:
                    out.write(file_bytes)
            except Exception as e:
                return jsonify({"error": f"Failed to write fallback image: {e}"}), 500

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

