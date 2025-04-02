import streamlit as st
import torch
from PIL import Image
import numpy as np
import requests
from io import BytesIO
from torchvision.transforms.functional import to_tensor, to_pil_image

# App setup
st.set_page_config(page_title="AnimeGAN2 Converter", layout="wide")
st.title("🎨 AnimeGAN2 Image Converter")

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model():
    try:
        model = torch.hub.load(
            'bryandlee/animegan2-pytorch:main',
            'generator',
            pretrained=True,
            device=device
        )
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def convert_to_anime(image, model):
    image = image.convert('RGB')
    img_tensor = to_tensor(image).unsqueeze(0) * 2 - 1
    with torch.no_grad():
        output = model(img_tensor.to(device)).cpu()
    output = (output.squeeze().permute(1, 2, 0).numpy() + 1) / 2 * 255
    return output.clip(0, 255).astype(np.uint8)

# Main app
model = load_model()
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    col1, col2 = st.columns(2)
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
    
    if st.button("Convert to Anime"):
        with st.spinner("Converting..."):
            anime_img = convert_to_anime(image, model)
            with col2:
                st.image(anime_img, caption="Anime Version", use_column_width=True)
                buf = BytesIO()
                Image.fromarray(anime_img).save(buf, format="JPEG")
                st.download_button(
                    "Download Result",
                    buf.getvalue(),
                    "anime_version.jpg",
                    "image/jpeg"
                )
