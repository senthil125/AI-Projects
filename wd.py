import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
from inference_sdk import InferenceHTTPClient

st.title("Crowd Weapon Detection")

# Confidence and overlap thresholds
CONFIDENCE_THRESHOLD = 0.50
OVERLAP_THRESHOLD = 0.45

# Roboflow SDK client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="6s4UDc7PrYIeBs0lbIkl"
)

# List of weapon-related classes
WEAPON_CLASSES = {'knife', 'gun', 'guns', 'hand', 'pistol', 'rifle', 'ruler', 'weapon'}

# Upload image through Streamlit
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg", "webp", "bmp", "tiff"])

if uploaded_file is not None:
    # Read and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save the image as a temporary file for inference
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file, format='JPEG')
        temp_file_path = temp_file.name

    # Create a copy for drawing bounding boxes
    result_image = image_np.copy()

    # Make prediction using the Inference SDK
    with st.spinner("Analyzing image..."):
        try:
            # Include confidence and overlap thresholds in the model_id string
            model_id_with_params = f"weapon-detection-cctv-v3-dataset/1?confidence={CONFIDENCE_THRESHOLD}&overlap={OVERLAP_THRESHOLD}"

            # Perform inference with the image file path
            result = CLIENT.infer(temp_file_path, model_id=model_id_with_params)

            # Track detected classes
            detected_classes = set()

            # Draw bounding boxes
            if "predictions" in result:
                for prediction in result["predictions"]:
                    cls_name = prediction["class"].lower()
                    conf = prediction["confidence"]

                    # Add detected class to the set
                    detected_classes.add(cls_name)

                    # Draw bounding boxes
                    x, y = prediction["x"], prediction["y"]
                    width, height = prediction["width"], prediction["height"]
                    
                    x1, y1 = int(x - width / 2), int(y - height / 2)
                    x2, y2 = int(x + width / 2), int(y + height / 2)

                    # Draw rectangle and label
                    color = (0, 255, 0) if cls_name in WEAPON_CLASSES else (255, 0, 0)
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(result_image, f'{cls_name}: {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Display image with bounding boxes
            st.image(result_image, caption="Detected Objects", use_column_width=True)

            # Detection summary
            if detected_classes == {'person'}:
                st.warning("No weapons detected in the image. Only people identified.")
            elif detected_classes & WEAPON_CLASSES:
                st.success("Weapon detected in the image!")
            else:
                st.warning("No weapons & persons detected in the image.")

            # Debugging: Display raw API response
            with st.expander("API Response Details"):
                st.json(result)

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

