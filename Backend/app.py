import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from rembg import remove # For background removal
from PIL import Image # For rembg compatibility
import io # To handle image bytes
import mediapipe as mp # For landmark detection (ready for try-on feature)
import time # For simulating processing time

# Initialize Flask app
# IMPORTANT: template_folder='.' tells Flask to look for HTML files (like index.html)
# in the SAME directory where app.py is located.
app = Flask(__name__, template_folder='.')

# Configure folders for static files and uploads
app.config['STATIC_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
app.config['MODELS_FOLDER'] = os.path.join(app.config['STATIC_FOLDER'], 'models')
app.config['UPLOADS_FOLDER'] = os.path.join(app.config['STATIC_FOLDER'], 'uploads')

# Create necessary folders if they don't exist
os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)
os.makedirs(app.config['UPLOADS_FOLDER'], exist_ok=True)

# --- MediaPipe Initialisation (for future/integrated try-on feature) ---
# These are initialized here but not directly used in the /api/process-image route
# as the current frontend doesn't send a model_id or expect try-on.
mp_face_mesh = mp.solutions.face_mesh
face_mesh_detector = mp_face_mesh.FaceMesh(
    static_image_mode=True, # For static images, not video stream
    max_num_faces=1,      # We only expect one face in the AI model image
    min_detection_confidence=0.5
)

mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1, # 0, 1, or 2. 1 is good balance for accuracy/speed
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Helper function for Face Mesh detection (for future earring placement) ---
def detect_face_landmarks(image_bgr):
    """Detects facial landmarks using MediaPipe Face Mesh."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh_detector.process(image_rgb)
    landmarks = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image_bgr.shape
            # Approximate earlobe positions (adjust these indices based on actual model and desired placement)
            # These are heuristic points. For precise placement, you'd need to fine-tune.
            left_ear_x, left_ear_y = int(face_landmarks.landmark[132].x * w), int(face_landmarks.landmark[132].y * h)
            right_ear_x, right_ear_y = int(face_landmarks.landmark[361].x * w), int(face_landmarks.landmark[361].y * h)
            landmarks.append({
                'left_ear': (left_ear_x, left_ear_y),
                'right_ear': (right_ear_x, right_ear_y)
            })
    return landmarks

# --- Helper function for Pose detection (for future necklace placement) ---
def detect_pose_landmarks(image_bgr):
    """Detects pose landmarks using MediaPipe Pose."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image_rgb)

    landmarks = []
    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark
        h, w, _ = image_bgr.shape
        
        # Key points for necklace: shoulders, nose (for head orientation)
        lm_left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        lm_right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lm_nose = pose_landmarks[mp_pose.PoseLandmark.NOSE]

        shoulder_mid_x = int(((lm_left_shoulder.x + lm_right_shoulder.x) / 2) * w)
        shoulder_mid_y = int(((lm_left_shoulder.y + lm_right_shoulder.y) / 2) * h)
        
        landmarks.append({
            'shoulder_mid': (shoulder_mid_x, shoulder_mid_y),
            'left_shoulder': (int(lm_left_shoulder.x * w), int(lm_left_shoulder.y * h)),
            'right_shoulder': (int(lm_right_shoulder.x * w), int(lm_right_shoulder.y * h)),
            'nose': (int(lm_nose.x * w), int(lm_nose.y * h))
        })
    return landmarks

# --- Helper function for overlaying images with alpha channel (for future try-on) ---
def overlay_image(background_img, overlay_img_rgba, x, y):
    """Overlays an RGBA image onto a BGR image at specified coordinates (center of overlay_img_rgba)."""
    h_bg, w_bg, _ = background_img.shape
    h_overlay, w_overlay, _ = overlay_img_rgba.shape

    # Calculate top-left corner for placement
    y1_overlay = y - h_overlay // 2
    x1_overlay = x - w_overlay // 2

    # Ensure coordinates are within bounds
    y1 = max(0, y1_overlay)
    x1 = max(0, x1_overlay)
    y2 = min(h_bg, y1_overlay + h_overlay)
    x2 = min(w_bg, x1_overlay + w_overlay)

    # Calculate corresponding slice of the overlay image
    overlay_slice_y1 = y1 - y1_overlay
    overlay_slice_x1 = x1 - x1_overlay
    overlay_slice_y2 = overlay_slice_y1 + (y2 - y1)
    overlay_slice_x2 = overlay_slice_x1 + (x2 - x1)

    overlay_cropped = overlay_img_rgba[overlay_slice_y1:overlay_slice_y2, overlay_slice_x1:overlay_slice_x2]

    if overlay_cropped.shape[0] == 0 or overlay_cropped.shape[1] == 0:
        return # Nothing to overlay if cropped out

    # Get alpha and RGB channels from overlay
    alpha_channel = overlay_cropped[:, :, 3] / 255.0
    rgb_overlay = overlay_cropped[:, :, :3]

    # Get the region of interest from the background
    roi = background_img[y1:y2, x1:x2]

    # Perform blending for each color channel
    for c in range(0, 3):
        background_img[y1:y2, x1:x2, c] = (
            alpha_channel * rgb_overlay[:, :, c] +
            (1.0 - alpha_channel) * roi[:, :, c]
        )

# --- Helper function for background removal ---
def clean_jewelry_image(image_bytes):
    """Removes background from an image using rembg."""
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    output_pil_image = remove(pil_image)
    
    # Convert back to bytes for further processing or saving
    img_byte_arr = io.BytesIO()
    output_pil_image.save(img_byte_arr, format='PNG') # PNG to preserve transparency
    return img_byte_arr.getvalue()


# --- Flask Routes ---

@app.route('/')
def serve_index():
    """Serves the main HTML page (index.html)."""
    # Flask will look for index.html in the directory specified by template_folder (which is '.' here)
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serves static files (like images from 'static/models' and 'static/uploads')."""
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

@app.route('/api/process-image', methods=['POST'])
def process_image_api():
    """
    Handles image upload, performs background removal, and simulates analysis results.
    This endpoint is designed for the current 'LuxeLens Analysis Tool' UI.
    """
    if 'jewelry_image' not in request.files:
        return jsonify({'error': 'No jewelry image provided.'}), 400

    jewelry_file = request.files['jewelry_image']

    try:
        start_time = time.time() # Start timing for processing time simulation

        # --- 1. Clean the Jewelry Image (Background Removal) ---
        jewelry_image_bytes = jewelry_file.read()
        cleaned_jewelry_bytes = clean_jewelry_image(jewelry_image_bytes)
        
        # Convert cleaned bytes to OpenCV format for potential future processing (e.g., defect detection)
        cleaned_jewelry_np = np.frombuffer(cleaned_jewelry_bytes, np.uint8)
        cleaned_jewelry_img_rgba = cv2.imdecode(cleaned_jewelry_np, cv2.IMREAD_UNCHANGED)

        if cleaned_jewelry_img_rgba is None or cleaned_jewelry_img_rgba.shape[2] != 4:
             return jsonify({'error': 'Could not process jewelry image or it lacks transparency.'}), 500

        # Save the original uploaded image for 'originalImage' display
        # IMPORTANT: Reset file pointer before reading again if you read it once (e.g., for clean_jewelry_image)
        jewelry_file.seek(0)
        original_filename = f"original_jewelry_{os.urandom(8).hex()}.jpg"
        original_filepath = os.path.join(app.config['UPLOADS_FOLDER'], original_filename)
        original_img_np = np.frombuffer(jewelry_file.read(), np.uint8)
        original_img = cv2.imdecode(original_img_np, cv2.IMREAD_COLOR)
        cv2.imwrite(original_filepath, original_img)


        # Save the cleaned image to return its URL
        cleaned_filename = f"cleaned_jewelry_{os.urandom(8).hex()}.png"
        cleaned_filepath = os.path.join(app.config['UPLOADS_FOLDER'], cleaned_filename)
        cv2.imwrite(cleaned_filepath, cleaned_jewelry_img_rgba) # Save as PNG to preserve alpha

        # --- 2. Simulate Analysis Results (Replace with real logic for hackathon) ---
        # For a hackathon, you'd implement your defect detection, authenticity calculation here.
        # For now, we'll use mock data and a simulated processing time.
        
        # Simulate some processing time
        time.sleep(2) # Simulate 2 seconds of work

        mock_defects = [
            {"type": "Minor scratch", "confidence": 87, "location": {"x": 45, "y": 30, "width": 15, "height": 8}, "severity": "low"},
            {"type": "Discoloration", "confidence": 92, "location": {"x": 60, "y": 55, "width": 20, "height": 12}, "severity": "medium"},
            {"type": "Inclusion", "confidence": 95, "location": {"x": 20, "y": 70, "width": 10, "height": 10}, "severity": "high"},
        ]
        
        end_time = time.time()
        processing_duration = round(end_time - start_time, 2) # Calculate actual processing time

        response_data = {
            "originalImage": f"/static/uploads/{original_filename}", # URL of the original uploaded image
            "cleanedImage": f"/static/uploads/{cleaned_filename}", # URL of the cleaned image
            "defects": mock_defects,
            "authenticityScore": 94,
            "processingTime": processing_duration
        }

        # --- Virtual Try-On Logic (Placeholder for future integration) ---
        # If you were to add Try-On:
        # - This endpoint would also receive a 'model_id'
        # - Load model_img = cv2.imread(os.path.join(app.config['MODELS_FOLDER'], f"{selected_model_id}.jpg"))
        # - Use MediaPipe (detect_face_landmarks / detect_pose_landmarks) on model_img
        # - Overlay cleaned_jewelry_img_rgba onto model_img using overlay_image function
        # - Save the final try-on image and return its URL.

        return jsonify(response_data)

    except Exception as e:
        print(f"Error during image processing: {e}")
        return jsonify({'error': f'An internal server error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    # Ensure your AI model images are in backend/static/models/
    # e.g., backend/static/models/model1.jpg, model2.jpg etc.
    print(f"Serving static files from: {app.config['STATIC_FOLDER']}")
    print(f"AI model images should be in: {app.config['MODELS_FOLDER']}")
    print(f"Uploaded/processed images will be saved to: {app.config['UPLOADS_FOLDER']}")
    app.run(debug=True, port=5000)