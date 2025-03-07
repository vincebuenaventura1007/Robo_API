import sys
import os
import base64
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ✅ Fix Unicode issues for Windows terminals
sys.stdout.reconfigure(encoding="utf-8")

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter API calls

# Roboflow API Configuration
ROBOFLOW_API_URL = "https://detect.roboflow.com/infer/workflows/masid-nert8/detect-count-and-visualize"
ROBOFLOW_API_KEY = "eWs6KSOlnWifknc0nP1U"

# Upload folder (temporarily stores images)
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask Roboflow API is running!"}), 200


@app.route("/api/detect", methods=["POST"])
def detect_image():
    if "image" not in request.files:
        print("❌ No image received")
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]

    # Save the uploaded image
    filename = secure_filename(image_file.filename)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image_file.save(image_path)

    print(f"✅ Received Image: {filename}")
    print(f"📂 Saved Image Path: {image_path}")

    # Convert image to Base64
    with open(image_path, "rb") as img:
        base64_image = base64.b64encode(img.read()).decode("utf-8")

    print("🔄 Converting Image to Base64...")

    # Send request to Roboflow API
    payload = {
        "api_key": ROBOFLOW_API_KEY,
        "inputs": {
            "image": {"type": "base64", "value": base64_image}
        },
    }

    print("📤 Sending image to Roboflow...")
    response = requests.post(ROBOFLOW_API_URL, json=payload)

    print(f"🔍 Roboflow Response Status: {response.status_code}")
    print(f"📊 Roboflow Response: {response.text}")

    if response.status_code == 200:
        try:
            data = response.json()

            # Debugging: Check the full response structure
            print(f"🔍 API Response Structure: {data}")

            # ✅ Extract predictions from the correct location
            outputs = data.get("outputs", [])
            if not outputs or not isinstance(outputs, list):
                return jsonify({"error": "Unexpected response format"}), 500

            first_output = outputs[0] if outputs else {}
            predictions_data = first_output.get("predictions", {})

            if "image" in predictions_data:
                predictions = predictions_data["predictions"]  # ✅ Correctly accessing predictions list
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
            return jsonify({"result": formatted_result}), 200
        except Exception as e:
            print(f"⚠️ Error processing API response: {e}")
            return jsonify({"error": "Error processing API response"}), 500
    else:
        return jsonify({"error": "Failed to get response from Roboflow", "response": response.text}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
