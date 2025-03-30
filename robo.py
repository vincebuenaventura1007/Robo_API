import sys
import os
import base64
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Fix Unicode issues for Windows terminals
sys.stdout.reconfigure(encoding="utf-8")

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter API calls

# Roboflow API Configuration
ROBOFLOW_API_URL = "https://detect.roboflow.com/infer/workflows/masid-nert8/detect-count-and-visualize"
ROBOFLOW_API_KEY = "eWs6KSOlnWifknc0nP1U"

# Allowed image formats
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# We only need to store processed images on disk now
PROCESSED_FOLDER = "processed"
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask Roboflow API is running!"}), 200

@app.route("/api/detect", methods=["POST"])
def detect_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]

    # Check if the file format is allowed
    if not allowed_file(image_file.filename):
        return jsonify({"error": "Unsupported file format. Please upload a PNG or JPG image."}), 400

    filename = secure_filename(image_file.filename)
    
    # Read file bytes directly from memory instead of saving to disk first
    try:
        file_bytes = image_file.read()
        base64_image = base64.b64encode(file_bytes).decode("utf-8")
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {e}"}), 500

    # Send request to Roboflow API
    payload = {
        "api_key": ROBOFLOW_API_KEY,
        "inputs": {
            "image": {"type": "base64", "value": base64_image}
        },
    }

    response = requests.post(ROBOFLOW_API_URL, json=payload, timeout=10)
    
    if response.status_code != 200:
        return jsonify({"error": "Failed to process image"}), 500

    try:
        data = response.json()

        # Extract processed image Base64
        processed_image_base64 = None
        if "outputs" in data and isinstance(data["outputs"], list) and len(data["outputs"]) > 0:
            first_output = data["outputs"][0]
            if "output_image" in first_output and isinstance(first_output["output_image"], dict):
                processed_image_base64 = first_output["output_image"].get("value", None)

        if not processed_image_base64:
            return jsonify({"error": "Processed image not found"}), 500

        # Save the processed image locally
        processed_filename = "processed_" + filename
        processed_image_path = os.path.join(app.config["PROCESSED_FOLDER"], processed_filename)
        with open(processed_image_path, "wb") as img_file:
            img_file.write(base64.b64decode(processed_image_base64))

        # Extract predictions safely
        outputs = data.get("outputs", [])
        if not outputs or not isinstance(outputs, list):
            return jsonify({"error": "Unexpected response format"}), 500

        first_output = outputs[0] if outputs else {}
        predictions_data = first_output.get("predictions", {})

        predictions = predictions_data.get("predictions", [])

        # Count occurrences of each class
        class_counts = {}
        for obj in predictions:
            if isinstance(obj, dict):
                class_name = obj.get("class", "Unknown")
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # Prepare formatted result
        if len(class_counts) == 1:
            detected_class = list(class_counts.keys())[0]
            count = class_counts[detected_class]
            formatted_result = f"{count} {detected_class}"
        else:
            formatted_result = {
                "ingredients": sum(class_counts.values()),
                "details": [{"count": count, "class": c} for c, count in class_counts.items()],
            }

        # Generate processed image URL
        processed_image_url = f"http://{request.host}/processed/{processed_filename}"

        return jsonify({
            "result": formatted_result,
            "processed_image_url": processed_image_url,
            "processed_image_base64": processed_image_base64
        }), 200

    except Exception as e:
        return jsonify({"error": f"Error processing API response: {e}"}), 500

@app.route("/processed/<filename>", methods=["GET"])
def serve_processed_image(filename):
    """Serve the processed image to Flutter."""
    return send_from_directory(app.config["PROCESSED_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
