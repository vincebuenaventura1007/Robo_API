import sys
import os
import base64
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ✅ Fix Unicode issues for Windows terminals
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

# Upload folder (temporarily stores images)
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask Roboflow API is running!"}), 200


@app.route("/api/detect", methods=["POST"])
def detect_image():
    if "image" not in request.files:
        print("❌ No image received")
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]

    # ✅ Check if the file format is allowed
    if not allowed_file(image_file.filename):
        print("❌ Unsupported image format")
        return jsonify({"error": "Unsupported file format. Please upload a PNG or JPG image."}), 400

    # Save the uploaded image
    filename = secure_filename(image_file.filename)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image_file.save(image_path)

    print(f"✅ Received Image: {filename}")
    print(f"📂 Saved Image Path: {image_path}")

    # Convert image to Base64
    try:
        with open(image_path, "rb") as img:
            base64_image = base64.b64encode(img.read()).decode("utf-8")
    except Exception as e:
        print(f"❌ Error encoding image to Base64: {e}")
        return jsonify({"error": "Failed to encode image"}), 500

    print("🔄 Converting Image to Base64...")

    # Send request to Roboflow API (with retry)
    payload = {
        "api_key": ROBOFLOW_API_KEY,
        "inputs": {
            "image": {"type": "base64", "value": base64_image}
        },
    }

    max_retries = 3
    response = None

    for attempt in range(max_retries):
        print(f"📤 Sending image to Roboflow... Attempt {attempt + 1}/{max_retries}")
        try:
            response = requests.post(ROBOFLOW_API_URL, json=payload, timeout=10)
            if response.status_code == 200:
                break  # Success, stop retrying
        except requests.exceptions.RequestException as e:
            print(f"⚠️ API Request Error: {e}")

    if response is None or response.status_code != 200:
        print(f"❌ Failed to get a valid response from Roboflow after {max_retries} attempts")
        return jsonify({"error": "Failed to process image"}), 500

    try:
        data = response.json()
        print(f"📊 Full API Response: {data}")  # ✅ Debugging full response

        # ✅ Extract processed image Base64
        processed_image_base64 = None
        if "outputs" in data and isinstance(data["outputs"], list) and len(data["outputs"]) > 0:
            first_output = data["outputs"][0]  # ✅ Extract first output
            if "output_image" in first_output and isinstance(first_output["output_image"], dict):
                processed_image_base64 = first_output["output_image"].get("value", None)

        if processed_image_base64:
            print(f"🖼️ Processed Image Base64 Length: {len(processed_image_base64)}")
        else:
            print("⚠️ Processed Image Base64 Not Found")

        # ✅ Save the processed image locally
        processed_filename = "processed_" + filename
        processed_image_path = os.path.join(app.config["PROCESSED_FOLDER"], processed_filename)
        with open(processed_image_path, "wb") as img_file:
            img_file.write(base64.b64decode(processed_image_base64))

        # ✅ Extract predictions safely
        outputs = data.get("outputs", [])
        if not outputs or not isinstance(outputs, list):
            return jsonify({"error": "Unexpected response format"}), 500

        first_output = outputs[0] if outputs else {}
        predictions_data = first_output.get("predictions", {})

        if "predictions" in predictions_data:
            predictions = predictions_data["predictions"]
        else:
            return jsonify({"error": "Predictions not found in response"}), 500

        print(f"📋 Raw Predictions: {predictions}")

        # ✅ Count occurrences of each class
        class_counts = {}
        for obj in predictions:
            if isinstance(obj, dict):  # Ensure obj is a dictionary
                class_name = obj.get("class", "Unknown")
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # ✅ If only one class is present, return formatted response
        if len(class_counts) == 1:
            detected_class = list(class_counts.keys())[0]
            count = class_counts[detected_class]
            formatted_result = f"{count} {detected_class}"
        else:
            # Otherwise, return full details
            formatted_result = {
                "ingredients": sum(class_counts.values()),
                "details": [{"count": count, "class": c} for c, count in class_counts.items()],
            }

        print(f"✅ Final Response: {formatted_result}")

        # Generate processed image URL
        processed_image_url = f"http://{request.host}/processed/{processed_filename}"

        return jsonify({
            "result": formatted_result,
            "processed_image_url": processed_image_url
        }), 200

    except Exception as e:
        print(f"⚠️ Error processing API response: {e}")
        return jsonify({"error": "Error processing API response"}), 500


@app.route("/processed/<filename>", methods=["GET"])
def serve_processed_image(filename):
    """Serve the processed image to Flutter."""
    return send_from_directory(app.config["PROCESSED_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
