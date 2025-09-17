
from flask import Flask, render_template, request, jsonify, Response

from ultralytics import YOLO
from PIL import Image
import base64
import yaml
from io import BytesIO
import os
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Load data.yaml
with open('data-baru.yaml', 'r') as f:
    yaml_data = yaml.safe_load(f)

class_names = yaml_data['names']
class_info = yaml_data['info']

# Load model
detector = YOLO('best-2.pt')

@app.route('/')
def index():
    return render_template('index.html', leaf_data=class_info)

@app.route('/detection')
def detection_page():
    return render_template('detection.html')

@app.route('/recommendation')
def recommendation_page():
    return render_template('recommendation.html')

@app.route('/upload-detection')
def upload_detection():
    return render_template('upload-detection.html')

@app.route('/webcam')
def webcam_detection():
    return render_template('webcam.html')

last_detection = []

def gen_frames():
    global last_detection
    cap = cv2.VideoCapture(0)  # webcam default
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            results = detector.predict(frame, verbose=False)
            detections = []

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes.xyxy):
                        x1, y1, x2, y2 = [int(x) for x in box]
                        class_id = int(boxes.cls[i].item())
                        class_name = class_names[class_id]

                        info = class_info.get(class_name, {})
                        components = info.get('components', [])
                        benefits = info.get('benefits', [])
                        recipes_raw = info.get('recipes', {})

                        converted_recipes = {}
                        for benefit, recipe in recipes_raw.items():
                            converted_recipes[benefit] = {
                                'bahan': recipe.get('ingredients', []),
                                'langkah': recipe.get('steps', [])
                            }

                        detections.append({
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2,
                            'class_name': class_name,
                            'components': components,
                            'benefits': benefits,
                            'recipes': converted_recipes
                        })

                        # ==== Gambar bounding box ====
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(frame, class_name, (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                        # ==== Tambahkan info components & benefits ====
                        text_y = y2 + 20
                        if components:
                            cv2.putText(frame, f"Comp: {', '.join(components[:3])}", 
                                        (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.5, (255, 255, 255), 1)
                            text_y += 20
                        if benefits:
                            cv2.putText(frame, f"Benefit: {', '.join(benefits[:2])}", 
                                        (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.5, (255, 255, 255), 1)

            last_detection = detections  # simpan hasil deteksi terakhir

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam_data')
def webcam_data():
    return jsonify({
        'status': 'success',
        'bounding_boxes': last_detection
    })

@app.route('/submit', methods=['POST'])
def submit_data():
    try:
        data = request.get_json()
        image_data = data.get('image')

        if not image_data:
            return jsonify({'status': 'error', 'message': 'No image data'}), 400

        # Decode base64 to image
        image_str = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_str)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Predict using YOLO model
        results = detector.predict(image)
        bounding_boxes = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes.xyxy):
                    x1, y1, x2, y2 = [int(x) for x in box]
                    class_id = int(boxes.cls[i].item())
                    class_name = class_names[class_id]

                    info = class_info.get(class_name, {})
                    components = info.get('components', [])
                    benefits = info.get('benefits', [])
                    recipes_raw = info.get('recipes', {})

                    # Convert key names for frontend compatibility
                    converted_recipes = {}
                    for benefit, recipe in recipes_raw.items():
                        converted_recipes[benefit] = {
                            'bahan': recipe.get('ingredients', []),
                            'langkah': recipe.get('steps', [])
                        }

                    bounding_boxes.append({
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'class_name': class_name,
                        'components': components,
                        'benefits': benefits,
                        'recipes': converted_recipes
                    })

        return jsonify({
            'status': 'success',
            'bounding_boxes': bounding_boxes
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route("/api/recommend")
def recommend():
    query = request.args.get("q", "").lower()
    if not query:
        return jsonify({"status": "error", "message": "Query kosong", "results": []})

    results = []
    for leaf_name, leaf_data in yaml_data["info"].items():
        match = [b for b in leaf_data.get("benefits", []) if query in b.lower()]
        if match:
            # ambil path gambar dari yaml
            gambar_path = leaf_data.get("gambar", "")
            # pastikan format URL cocok dengan Flask static
            if gambar_path.startswith("static/"):
                gambar_url = "/" + gambar_path
            else:
                gambar_url = gambar_path

            results.append({
                "leaf_name": leaf_name,
                "gambar": gambar_url,   # <---- tambahin ini
                "components": leaf_data.get("components", []),
                "benefits": leaf_data.get("benefits", []),
                "recipes": leaf_data.get("recipes", {})
            })

    return jsonify({"status": "success", "results": results})
# app.py

@app.route('/upload-detection', methods=['POST'])
def upload_image():
    try:
        if 'file' in request.files:
            file = request.files['file']
        else:
            return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400

        image = Image.open(file.stream).convert("RGB")
        results = detector.predict(image)
        bounding_boxes = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes.xyxy):
                    x1, y1, x2, y2 = [int(x) for x in box]
                    class_id = int(boxes.cls[i].item())
                    class_name = class_names[class_id]
                    info = class_info.get(class_name, {})
                    components = info.get('components', [])
                    benefits = info.get('benefits', [])
                    recipes_raw = info.get('recipes', {})

                    converted_recipes = {}
                    for benefit, recipe in recipes_raw.items():
                        converted_recipes[benefit] = {
                            'bahan': recipe.get('ingredients', []),
                            'langkah': recipe.get('steps', [])
                        }

                    bounding_boxes.append({
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'class_name': class_name,
                        'components': components,
                        'benefits': benefits,
                        'recipes': converted_recipes
                    })

        return jsonify({
            'status': 'success',
            'bounding_boxes': bounding_boxes
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
