import streamlit as st
import tempfile
import cv2

from predict import predict_output
from explain import generate_gradcam


st.title("Pneumonia Detection System with Explainable AI")
st.write(
    "Upload a chest X-ray image for AI-assisted pneumonia detection."
)

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded X-ray",
        use_container_width=True
    )
    
    st.markdown("""
        <style>
        div.stButton {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        div.stButton > button:first-child {
            background-color: #c2185b;
            color: white;
            width: 200px;
            height: 60px;
            font-size: 30px;
            font-weight: bold;
            border-radius: 20px;
            border:none;
        }
            
        div.stButton> button:first-child:hover{
            background-color: #1565c0;       
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    
    if st.button("Predict"):
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        result = predict_output(temp_path)


        st.subheader("Prediction")
        st.write(
            f"Predicted Class: "
            f"{result['predicted_class']}"
        )
        st.write(
            f"Likelihood: "
            f"{result['likelihood']}"
        )
        st.write(
            f"Confidence: "
            f"{result['confidence']:.2f}%"
        )

        st.subheader("Grad-CAM Explainability")

        gradcam_img = generate_gradcam(temp_path)
        gradcam_img = cv2.cvtColor(
            gradcam_img,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            gradcam_img,
            caption="Highlighted Regions in Red and Yellow",
            use_container_width=True
        )


st.warning(
    "This system is for AI-assisted analysis only "
    "and not a replacement for professional "
    "medical diagnosis."
)