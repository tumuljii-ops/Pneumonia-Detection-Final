import streamlit as st

import requests

st.title(
    "Pneumonia Detection"
)

st.write(
    "Upload a Chest X-Ray Image"
)

uploaded_file = st.file_uploader(

    "Choose Image",

    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)

if uploaded_file:

    st.image(
        uploaded_file,
        use_container_width=True
    )

    if st.button("Predict"):

        files = {

            "file": uploaded_file
        }

        response = requests.post(

            "https://pneumonia-detection-final-2.onrender.com",

            files=files
        )

        result = response.json()

        st.success(

            f"Prediction: "
            f"{result['prediction']}"
        )

        st.info(

            f"Confidence: "
            f"{result['confidence']}%"
        )