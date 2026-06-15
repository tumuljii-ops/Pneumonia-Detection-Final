import streamlit as st
import requests

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁"
)

st.title("🫁 Pneumonia Detection")

st.write(
    "Upload a Chest X-Ray image and get prediction."
)

uploaded_file = st.file_uploader(
    "Choose Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):

        files = {
            "file": uploaded_file
        }

        try:

            response = requests.post(
                "https://pneumonia-detection-final-2.onrender.com/predict",
                files=files,
                timeout=120
            )

            result = response.json()

            if "prediction" in result:

                st.success(
                    f"Prediction: {result['prediction']}"
                )

                st.info(
                    f"Confidence: {result['confidence']}%"
                )

            else:

                st.error(
                    f"Unexpected Response: {result}"
                )

        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )