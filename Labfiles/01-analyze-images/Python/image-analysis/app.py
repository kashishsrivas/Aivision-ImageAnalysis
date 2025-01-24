import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError


# Load environment variables
load_dotenv()
ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
ai_key = os.getenv('AI_SERVICE_KEY')

# Authenticate Azure AI Vision client
cv_client = ImageAnalysisClient(
    endpoint=ai_endpoint,
    credential=AzureKeyCredential(ai_key)
)


def analyze_image(image_file):
    """Analyze the image and display results."""
    try:
        # Read image data
        with open(image_file, "rb") as f:
            image_data = f.read()

        # Call Azure AI Vision service
        result = cv_client.analyze(
            image_data=image_data,
            visual_features=[
                VisualFeatures.CAPTION,
                VisualFeatures.DENSE_CAPTIONS,
                VisualFeatures.TAGS,
                VisualFeatures.OBJECTS,
                VisualFeatures.PEOPLE
            ]
        )

        st.subheader("Analysis Results")

        # Display Caption
        if result.caption is not None:
            st.markdown(f"*Caption:* {result.caption.text} (Confidence: {result.caption.confidence * 100:.2f}%)")

        # Display Dense Captions
        if result.dense_captions is not None:
            st.markdown("*Dense Captions:*")
            for caption in result.dense_captions.list:
                st.write(f"- {caption.text} (Confidence: {caption.confidence * 100:.2f}%)")

        # Display Tags
        if result.tags is not None:
            st.markdown("*Tags:*")
            tags = [f"{tag.name} (Confidence: {tag.confidence * 100:.2f}%)" for tag in result.tags.list]
            st.write(", ".join(tags))

        # Display Objects in Image
        if result.objects is not None:
            image = Image.open(image_file)
            draw = ImageDraw.Draw(image)
            st.markdown("*Objects in Image:*")
            for obj in result.objects.list:
                st.write(f"- {obj.tags[0].name} (Confidence: {obj.tags[0].confidence * 100:.2f}%)")
                r = obj.bounding_box
                draw.rectangle([(r.x, r.y), (r.x + r.width, r.y + r.height)], outline="cyan", width=3)

            st.image(image, caption="Annotated Image with Detected Objects")

        # Display People in Image
        if result.people is not None:
            image = Image.open(image_file)
            draw = ImageDraw.Draw(image)
            st.markdown("*People in Image:*")
            for person in result.people.list:
                r = person.bounding_box
                draw.rectangle([(r.x, r.y), (r.x + r.width, r.y + r.height)], outline="yellow", width=3)

            st.image(image, caption="Annotated Image with Detected People")

    except HttpResponseError as e:
        st.error(f"Error: {e.reason}. Message: {e.error.message}")


# Streamlit App Layout
st.title("Azure AI Vision - Image Analysis")
st.write("Upload an image to analyze it using Azure AI Vision.")

# File Upload
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.read())

    st.image("uploaded_image.jpg", caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze Image"):
        analyze_image("uploaded_image.jpg")