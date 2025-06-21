import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os

# --- Model Architecture & Constants ---
# This must match the training script exactly.
LATENT_DIM = 100
NUM_CLASSES = 10
EMBEDDING_DIM = 100
IMAGE_SIZE = 28
CHANNELS_IMG = 1
MODEL_PATH = "cgan_generator.pth"

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, embedding_dim):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim + embedding_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, IMAGE_SIZE * IMAGE_SIZE * CHANNELS_IMG),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_embedding(labels)
        x = torch.cat([z, c], 1)
        output = self.model(x)
        return output.view(-1, CHANNELS_IMG, IMAGE_SIZE, IMAGE_SIZE)


@st.cache_resource
def load_generator_model():
    """Loads the pre-trained generator model from disk."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at '{MODEL_PATH}'.")
        return None
    
    device = torch.device("cpu")
    generator = Generator(LATENT_DIM, NUM_CLASSES, EMBEDDING_DIM).to(device)
    generator.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    generator.eval()
    return generator

# --- UI Configuration ---
st.set_page_config(
    page_title="Digit Generator",
    layout="wide",
    initial_sidebar_state="expanded"  # This is the line that was added
)

st.title("✍️ Handwritten Digit Generation Web App")
st.markdown(
    """
    This app generates 5 unique images of a selected handwritten digit (0-9) 
    using a Conditional Generative Adversarial Network (CGAN).
    """
)

# --- Sidebar ---
st.sidebar.header("Controls")
selected_digit = st.sidebar.selectbox("Select a digit to generate:", options=list(range(10)), index=7)
generate_button = st.sidebar.button("Generate Images", type="primary", use_container_width=True)

# --- Main Content ---
generator = load_generator_model()

if generator:
    st.header(f"Generated Images for Digit: {selected_digit}")

    if generate_button:
        with st.spinner(f"Generating 5 different images of digit '{selected_digit}'..."):
            
            num_images_to_generate = 5
            cols = st.columns(num_images_to_generate)
            
            for i in range(num_images_to_generate):
                z = torch.randn(1, LATENT_DIM)
                label = torch.LongTensor([selected_digit])
                
                with torch.no_grad():
                    generated_image_tensor = generator(z, label)
                
                image_np = generated_image_tensor.cpu().numpy().squeeze()
                image_np = (image_np + 1) / 2.0
                
                with cols[i]:
                    st.image(image_np, use_column_width=True, caption=f"Image #{i+1}")
    else:
        st.info("Select a digit and click 'Generate Images' in the sidebar.")
else:
    st.warning("Could not load the generator model.")
