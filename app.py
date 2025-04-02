import streamlit as st
import torch
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import time
from torchvision.transforms.functional import to_tensor

# Configure page
st.set_page_config(
    page_title="AnimeGAN2 Converter",
    page_icon="🎨",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        padding: 2rem;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .stDownloadButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .title {
        text-align: center;
        color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="title">🎨 Anime Style Converter</h1>', unsafe_allow_html=True)
st.markdown("""
Transform your photos into beautiful anime artwork using AI! 
Upload any portrait or landscape photo and see it transformed into anime style.
""")

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.sidebar.info(f"Running on: {device}")

@st.cache_resource
def load_model():
    try:
        # Try loading face paint model first
        model = torch.hub.load(
            'bryandlee/animegan2-pytorch:main',
            'generator',
            pretrained=True,
            device=device
        )
        st.sidebar.success("Loaded face_paint_512_v2 model")
        return model
    except Exception as e:
        st.sidebar.warning(f"Primary model failed: {str(e)}")
        try:
            # Fallback to celeba model
            model = torch.hub.load(
                'bryandlee/animegan2-pytorch:main',
                'generator',
                pretrained='celeba_distill',
                device=device
            )
            st.sidebar.warning("Loaded celeba_distill as fallback")
            return model
        except Exception as e:
            st.sidebar.error(f"Failed to load all models: {str(e)}")
            return None

def preprocess(image, device):
    image = to_tensor(image).unsqueeze(0) * 2 - 1
    return image.to(device)

def convert_to_anime(image, model):
    try:
        # Convert to RGB if not already
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert('RGB')
        
        # Preprocess and convert
        input_image = preprocess(image, device)
        
        with torch.no_grad():
            output_image = model(input_image).cpu()
        
        # Post-process output
        output_image = output_image.squeeze().permute(1, 2, 0).numpy()
        output_image = (output_image + 1) / 2 * 255
        output_image = output_image.clip(0, 255).astype(np.uint8)
        
        return output_image
    
    except Exception as e:
        st.error(f"Conversion error: {str(e)}")
        return None

# Main app flow
model = load_model()

uploaded_file = st.file_uploader(
    "Choose an image to convert", 
    type=["jpg", "jpeg", "png"],
    help="For best results, use clear photos with good lighting"
)

if uploaded_file is not None and model is not None:
    # Display original image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    
    # Convert button
    if st.button("✨ Convert to Anime Style", use_container_width=True):
        with st.spinner("Working magic... This may take 10-30 seconds"):
            start_time = time.time()
            anime_image = convert_to_anime(image, model)
            processing_time = time.time() - start_time
            
            if anime_image is not None:
                with col2:
                    st.subheader("Anime Version")
                    st.image(anime_image, use_column_width=True)
                    
                    # Prepare download
                    buf = BytesIO()
                    Image.fromarray(anime_image).save(buf, format="JPEG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="⬇️ Download Anime Image",
                        data=byte_im,
                        file_name="anime_version.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
                
                st.success(f"Conversion completed in {processing_time:.1f} seconds!")
            else:
                st.error("Conversion failed. Please try another image.")

elif model is None:
    st.error("Failed to load the AI model. Please try again later or contact support.")
