import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from pdf2image import convert_from_bytes
import json

# Set the Teachable Machine model URL
model_url = "https://teachablemachine.withgoogle.com/models/BQ18sMdQs/"

# Define the Streamlit app
def main():
    # Set the page title
    st.title("Heart Attack Predictor")

    # Create a file uploader
    uploaded_file = st.file_uploader("Choose a file...", type=["jpg", "jpeg", "png", "pdf"])

    # Check if a file is uploaded
    if uploaded_file is not None:
        # Check if the file is a PDF
        if uploaded_file.type == 'application/pdf':
            # Convert PDF to images
            images = convert_from_bytes(uploaded_file.read())

            # Process each image
            for i, image in enumerate(images):
                # Display the uploaded image
                st.image(image, caption=f"Page {i+1}", use_column_width=True)

                # Convert the image to bytes
                img_bytes = BytesIO()
                image.save(img_bytes, format='JPEG')
                img_bytes.seek(0)

                # Send a POST request to the Teachable Machine model
                response = requests.post(model_url, files={'file': img_bytes})

                # Check if the request was successful
                if response.status_code == 200:
                    try:
                        # Get the prediction result
                        result = response.json()

                        # Display the prediction
                        st.header(f"Prediction - Page {i+1}")
                        for prediction in result['predictions']:
                            class_name = prediction['label']
                            confidence = prediction['confidence']
                            st.write(f"{class_name}: {confidence:.2f}")
                    except json.JSONDecodeError as e:
                        st.error(f"Error decoding JSON response: {str(e)}")
                        st.text(response.text)  # Print the response content for debugging

        else:
            # Load the image
            image = Image.open(uploaded_file)

            # Display the uploaded image
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Convert the image to bytes
            img_bytes = uploaded_file.read()

            # Send a POST request to the Teachable Machine model
            response = requests.post(model_url, files={'file': img_bytes})

            # Check if the request was successful
            if response.status_code == 200:
                try:
                    # Get the prediction result
                    result = response.json()

                    # Display the prediction
                    st.header("Prediction")
                    for prediction in result['predictions']:
                        class_name = prediction['label']
                        confidence = prediction['confidence']
                        st.write(f"{class_name}: {confidence:.2f}")
                except json.JSONDecodeError as e:
                    st.error(f"Error decoding JSON response: {str(e)}")
                    st.text(response.text)  # Print the response content for debugging

# Run the app
if __name__ == '__main__':
    main()
