# src/frontend/app.py
import streamlit as st
from PIL import Image
import requests
import os
from pathlib import Path  # Import Path

# Relative import for config.py to access IMAGES_DIR
import config


# Define the backend URL (adjust if your FastAPI app runs elsewhere)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(layout="wide")

def fetch_image_options():
    """Fetches the list of available image names from the backend."""
    try:
        response = requests.get(f"{BACKEND_URL}/images")
        response.raise_for_status()  # Raise an exception for HTTP errors
        all_images = response.json().get("images", [])
        # Filter images based on allowed_indices.  Crucially important.
        allowed_indices = [109, 120, 106, 25, 26, 30, 35, 238]

        # Load captions.txt to get the correct mapping.
        import pandas as pd
        captions_df = pd.read_csv(config.CAPTIONS_PATH) # Use the path from config.py
        val_df = captions_df.iloc[int(0.9 * len(captions_df)):]
        unq_val_imgs = val_df[['image']].drop_duplicates()
        allowed_images = [unq_val_imgs.iloc[i]['image'] for i in allowed_indices if i < len(unq_val_imgs)] # prevent index error
        return allowed_images

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching image list from backend: {e}")
        return []

st.title("🖼️ Flickr8k Image Captioning 💬")
st.markdown("""
Welcome to the Image Captioning application!
Select an image from the dropdown below to view its actual captions and a machine-generated prediction.
This system uses a Transformer-based model to understand the content of the image and describe it in words.
""")

image_options = fetch_image_options()

if not image_options:
    st.warning("No images available to select. Please ensure the backend is running and configured correctly.")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Select Image")
        selected_image_name = st.selectbox(
            "Choose an image from the Flickr8k validation set:",
            options=image_options,
            help="These images are from the pre-defined validation split."
        )

    if selected_image_name:
        # Construct the absolute path to the image file using config.IMAGES_DIR
        
        image_path = config.IMAGES_DIR
        image_path = image_path+"/"+selected_image_name
        print(f"Image path is {image_path}") # For Debugging
        image_path = Path(config.IMAGES_DIR) / selected_image_name


        with col1:
            if image_path.exists():
                try:
                    image = Image.open(image_path).convert("RGB")
                    st.image(image, caption=f"Selected: {selected_image_name}", use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading image {selected_image_name}: {e}")
            else:
                st.error(f"Image file not found at: {image_path}. Please check the IMAGES_DIR configuration and ensure images are present.  The path was constructed using IMAGES_DIR from config.py")

        with col2:
            st.subheader("📝 Captions")
            if st.button("✨ Generate Caption", key="generate_caption_button", use_container_width=True):
                with st.spinner("🤖 Generating caption..."):
                    try:
                        payload = {"image_name": selected_image_name}
                        response = requests.post(f"{BACKEND_URL}/predict", json=payload)
                        response.raise_for_status()
                        data = response.json()

                        st.markdown("#### Predicted Caption:")
                        st.success(f"> {data['predicted_caption']}")

                        st.markdown("#### Actual Captions:")
                        if data['actual_captions']:
                            for caption in data['actual_captions']:
                                st.info(f"- {caption}")
                        else:
                            st.write("No actual captions found for this image.")

                    except requests.exceptions.RequestException as e:
                        st.error(f"Error communicating with backend: {e}")
                        if hasattr(e, 'response') and e.response is not None:
                            try:
                                error_detail = e.response.json().get("detail", e.response.text)
                                st.error(f"Backend error detail: {error_detail}")
                            except ValueError:  # if response is not JSON
                                st.error(f"Backend response: {e.response.text}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
    else:
        st.info("Please select an image to see its captions.")

st.markdown("---")
st.markdown("Built with FastAPI & Streamlit.")
# src/frontend/app.py

# import streamlit as st
# from PIL import Image
# import requests
# import os

# # Local config module, which defines IMAGES_DIR as a pathlib.Path object
# import config

# # Define the backend URL (can be overridden with an environment variable)
# BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# # Set Streamlit page configuration
# st.set_page_config(layout="wide")

# def fetch_image_options():
#     """Fetches the list of available image names from the backend."""
#     try:
#         response = requests.get(f"{BACKEND_URL}/images")
#         response.raise_for_status()
#         return response.json().get("images", [])
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error fetching image list from backend: {e}")
#         return []

# # Page Title and Description
# st.title("🖼️ Flickr8k Image Captioning 💬")
# st.markdown("""
# Welcome to the Image Captioning application!  
# Select an image from the dropdown below to view its actual captions and a machine-generated prediction.

# This system uses a Transformer-based model to understand the content of the image and describe it in words.
# """)

# # Fetch image list from backend
# image_options = fetch_image_options()

# if not image_options:
#     st.warning("No images available to select. Please ensure the backend is running and configured correctly.")
# else:
#     col1, col2 = st.columns(2)

#     with col1:
#         st.subheader("📷 Select Image")
#         selected_image_name = st.selectbox(
#             "Choose an image from the Flickr8k validation set:",
#             options=image_options,
#             help="These images are from the pre-defined validation split."
#         )

#     if selected_image_name:
#         image_path = config.IMAGES_DIR / selected_image_name

#         with col1:
#             if image_path.exists():
#                 try:
#                     image = Image.open(image_path).convert("RGB")
#                     st.image(image, caption=f"Selected: {selected_image_name}", use_column_width=True)
#                 except Exception as e:
#                     st.error(f"Error loading image {selected_image_name}: {e}")
#             else:
#                 st.error(f"Image not found at: {image_path}. Please check IMAGES_DIR configuration.")

#         with col2:
#             st.subheader("📝 Captions")

#             if st.button("✨ Generate Caption", key="generate_caption_button", use_container_width=True):
#                 with st.spinner("🤖 Generating caption..."):
#                     try:
#                         payload = {"image_name": selected_image_name}
#                         response = requests.post(f"{BACKEND_URL}/predict", json=payload)
#                         response.raise_for_status()
#                         data = response.json()

#                         st.markdown("#### Predicted Caption:")
#                         st.success(f"> {data['predicted_caption']}")

#                         st.markdown("#### Actual Captions:")
#                         if data['actual_captions']:
#                             for caption in data['actual_captions']:
#                                 st.info(f"- {caption}")
#                         else:
#                             st.info("No actual captions found for this image.")

#                     except requests.exceptions.RequestException as e:
#                         st.error(f"Error communicating with backend: {e}")
#                         if e.response is not None:
#                             try:
#                                 error_detail = e.response.json().get("detail", e.response.text)
#                                 st.error(f"Backend error detail: {error_detail}")
#                             except ValueError:
#                                 st.error(f"Backend response: {e.response.text}")
#                     except Exception as e:
#                         st.error(f"An unexpected error occurred: {e}")
#     else:
#         st.info("Please select an image to see its captions.")

# st.markdown("---")
# st.markdown("Built with FastAPI & Streamlit.")
