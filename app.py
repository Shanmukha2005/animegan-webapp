import streamlit as st
import torch
from PIL import Image
import requests
from torchvision.transforms.functional import to_tensor

# Load model
@st.cache_resource
def load_model():
    model = torch.hub.load('bryandlee/animegan2-pytorch:main', 'generator', pretrained=False, device='cpu')
    model.load_state_dict(torch.load('face_paint_512_v2.pt', map_location='cpu'))
    model.eval()
    return model

# Process image
def convert_to_anime(image, model):
    image = to_tensor(image).unsqueeze(0) * 2 - 1
    with torch.no_grad():
        output = model(image).cpu().squeeze().permute(1, 2, 0).numpy()
        output = ((output + 1) / 2 * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(output)

# Streamlit UI
st.title("AnimeGAN2: Convert Your Images to Anime Style!")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    model = load_model()
    anime_image = convert_to_anime(image, model)
    st.image(anime_image, caption="Anime Image", use_column_width=True)
