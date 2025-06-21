import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os

# --- Model Architecture & Constants ---
# The model architecture must be defined here to reconstruct the model
# before loading its saved state (weights). These constants must exactly
# match the ones used in the training script.

LATENT_DIM = 100
NUM_CLASSES = 10
EMBEDDING_DIM = 100
IMAGE_SIZE = 28
CHANNELS_IMG = 1
MODEL_PATH = "cgan_generator.pth"

class Generator(nn.Module):
    """
    Generator Network (G).
    This class definition is required to load the pre-trained weights.
    """
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
            nn.Tanh() # Tanh activation scales output to [-1, 1]
        )

    def forward(self, z, labels):
        # Concatenate latent vector and label embedding
        c = self.label_embedding(labels)
        x = torch.cat([z, c], 1)
        output = self.model(x)
        return output.view(-1, CHANNELS_IMG, IMAGE_SIZE, IMAGE_SIZE)


# --- Streamlit Application ---

# Function to load the model.
# @st.cache_resource is used to load the model only once and cache it.
@st.cache_resource
def load_generator_model():
    """Loads the pre-trained generator model from disk."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at '{MODEL_PATH}'. Please ensure it's in the same directory as the app.")
        return None
    
    # Run model on CPU for inference, which is standard for deployment
    device = torch.device("cpu") 
    
    # Instantiate the model
    generator = Generator(LATENT_DIM, NUM_CLASSES, EMBEDDING_DIM).to(device)
    
    # Load the saved weights. map_location ensures it works on CPU
    generator.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    # Set the model to evaluation mode (disables dropout, etc.)
    generator.eval() 
    return generator

# --- UI Configuration ---

# Set page title and layout
st.set_page_config(page_title="Digit Generator", layout="wide")

# App Title and Description
st.title("✍️ Handwritten Digit Generation Web App")
st.markdown(
    """
    This application generates images of handwritten digits (0-9) using a 
    Conditional Generative Adversarial Network (CGAN) trained on the MNIST dataset.
    
    **Instructions:**
    1.  Use the dropdown menu in the sidebar to select a digit.
    2.  Click the **Generate Images** button.
    3.  The app will generate and display 5 unique images for the selected digit below.
    """
)

# --- Sidebar for User Controls ---
st.sidebar.header("Controls")
selected_digit = st.sidebar.selectbox(
    "Select a digit to generate:",
    options=list(range(10)),
    index=7 # Default to 7 for a nice-looking digit
)

generate_button = st.sidebar.button("Generate Images", type="primary", use_container_width=True)

# --- Main Content Area ---

# Load the generator model
generator = load_generator_model()

# Ensure the model is loaded before proceeding
if generator:
    st.header(f"Generated Images for Digit: {selected_digit}")

    # The generation logic runs only when the button is clicked
    if generate_button:
        # Display a spinner to indicate that work is being done
        with st.spinner(f"Generating 5 images of digit '{selected_digit}'..."):
            
            num_images_to_generate = 5
            
            # Create columns to display images side-by-side
            cols = st.columns(num_images_to_generate)
            
            for i in range(num_images_to_generate):
                # 1. Create a random noise vector (the source of variation)
                # This must be done inside the loop to get a unique image each time
                z = torch.randn(1, LATENT_DIM)
                
                # 2. Create the label for the selected digit
                label = torch.LongTensor([selected_digit])
                
                # 3. Generate the image using the model
                # torch.no_grad() is used for efficiency during inference
                with torch.no_grad():
                    generated_image_tensor = generator(z, label)
                
                # 4. Post-process the image for display
                # Move tensor to CPU, convert to numpy, and remove extra dimensions
                image_np = generated_image_tensor.cpu().numpy().squeeze()
                
                # Un-normalize from [-1, 1] (Tanh output) to [0, 1] range for display
                image_np = (image_np + 1) / 2.0
                
                # 5. Display the image in its column
                with cols[i]:
                    st.image(
                        image_np,
                        use_container_width=True,
                        caption=f"Image #{i+1}"
                    )
    else:
        # Initial message before the button is clicked
        st.info("Select a digit and click 'Generate Images' in the sidebar to start.")
else:
    # Display a warning if the model couldn't be loaded
    st.warning("Could not load the generator model. The app is non-functional.")
