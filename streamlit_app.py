import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Cache the VGG16 model as a resource
@st.cache_resource
def load_vgg16_model():
    model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    # Modifying the model structure to exclude the final classification layer, enabling access to the model's output features
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    return model

vgg16_model = load_vgg16_model()  # Create VGG16 model

# Load the trained model
model = load_model('mymodel.h5')

# Load the tokenizer from a file
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Streamlit app title and description
st.title("Image Caption Generator")
st.markdown("Upload an image, and this app will generate a caption for it using pre-trained VGG16 and customized LSTM model.")

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Process the uploaded image
if uploaded_image is not None:
    st.subheader("Uploaded Image")
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Generated Caption")
    
    # Display loading spinner while processing
    with st.spinner("Generating caption..."):
        # Load and preprocess the image
        image = load_img(uploaded_image, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        # Extract image features using VGG16
        image_features = vgg16_model.predict(image, verbose=0)

        # Maximum caption length
        max_caption_length = 34
        
        # Define a function to get the word from its index
        def get_word_from_index(index, tokenizer):
            return next((word for word, idx in tokenizer.word_index.items() if idx == index), None)

        # Generate a caption using the model
        def predict_caption(model, image_features, tokenizer, max_caption_length):
            caption = "startseq"
            for _ in range(max_caption_length):
                sequence = tokenizer.texts_to_sequences([caption])[0]
                sequence = pad_sequences([sequence], maxlen=max_caption_length)
                yhat = model.predict([image_features, sequence], verbose=0)
                predicted_index = np.argmax(yhat)
                predicted_word = get_word_from_index(predicted_index, tokenizer)
                caption += " " + predicted_word
                if predicted_word is None or predicted_word == "endseq":
                    break
            return caption

        # Generate a caption
        generated_caption = predict_caption(model, image_features, tokenizer, max_caption_length)

        # Remove startseq and endseq from the generated caption
        generated_caption = generated_caption.replace("startseq", "").replace("endseq", "")

    # Display the generated caption with custom styling
    st.markdown(
        f'<div style="border-left: 6px solid #ccc; padding: 5px 20px; margin-top: 20px;">'
        f'<p style="font-style: italic;">“{generated_caption}”</p>'
        f'</div>',
        unsafe_allow_html=True
    )
