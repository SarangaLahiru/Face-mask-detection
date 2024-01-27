import cv2
import streamlit as st
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# Load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

class FaceMaskDetector(VideoTransformerBase):
    def transform(self, frame):
        # Resize the frame to 224x224 and preprocess it for the model
        resized_frame = cv2.resize(frame, (224, 224))
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        resized_frame = img_to_array(resized_frame)
        resized_frame = preprocess_input(resized_frame)
        resized_frame = np.expand_dims(resized_frame, axis=0)

        # Predict if the person is wearing a mask or not
        preds = maskNet.predict(resized_frame)

        # Unpack the prediction
        (mask, withoutMask) = preds[0]

        # Determine the class label and color for the bounding box
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # Display the label and bounding box rectangle on the frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return frame

def main():
    st.title("Real-time Face Mask Detection")

    # Create a unique key for webrtc_streamer
    webrtc_key = "example_face_mask_detection"

    # Use webrtc_streamer to capture video from the webcam and apply face mask detection
    webrtc_streamer(
        key=webrtc_key,
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=FaceMaskDetector,
    )

if __name__ == "__main__":
    main()
import cv2
import streamlit as st
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# Load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

class FaceMaskDetector(VideoTransformerBase):
    def transform(self, frame):
        # Resize the frame to 224x224 and preprocess it for the model
        resized_frame = cv2.resize(frame, (224, 224))
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        resized_frame = img_to_array(resized_frame)
        resized_frame = preprocess_input(resized_frame)
        resized_frame = np.expand_dims(resized_frame, axis=0)

        # Predict if the person is wearing a mask or not
        preds = maskNet.predict(resized_frame)

        # Unpack the prediction
        (mask, withoutMask) = preds[0]

        # Determine the class label and color for the bounding box
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # Display the label and bounding box rectangle on the frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return frame

def main():
    st.title("Real-time Face Mask Detection")

    # Create a unique key for webrtc_streamer
    webrtc_key = "example_face_mask_detection"

    # Check if the webrtc component with the given key is already in the state
    if st.webrtc_state is not None and webrtc_key in st.webrtc_state:
        # If the webrtc component is already in the state, use it
        webrtc_result = st.webrtc(webrtc_key)
    else:
        # If the webrtc component is not in the state, create it
        webrtc_result = webrtc_streamer(
            key=webrtc_key,
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=FaceMaskDetector,
        )

    # Use the webrtc_result
    if webrtc_result:
        # Display the result or any additional content here
        st.write("WebRTC component created successfully.")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
