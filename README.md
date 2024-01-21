# Image Caption Generator

This project involves building an Image Caption Generator using transfer learning with a pre-trained VGG16 model and a customized LSTM neural network. The project consists of two main components:

1. **Model Training (`model.ipynb`):**
   - The model is constructed using transfer learning with the pre-trained VGG16 model as the encoder.
   - A customized LSTM neural network is used as the decoder to generate descriptive captions for input images.
   - The attention mechanism is applied to improve the model's ability to focus on relevant parts of the image while generating captions.
   - The model is trained using a dataset of images and corresponding captions.
   - The training script includes code for visualizing the model architecture, training the model, and saving/loading the trained model.

2. **Web Application Deployment (`streamlit_app.py`):**
   - The Streamlit web application provides a user-friendly interface for generating image captions.
   - Users can upload an image, and the application uses the trained model to generate a descriptive caption for the uploaded image.
   - The application utilizes a pre-trained VGG16 model for image feature extraction and a custom LSTM model for caption generation.
   - The application provides a seamless experience with one-click caption generation.

## Model Architecture
The LSTM neural network is designed with an attention mechanism for improved caption generation. The architecture includes an encoder-decoder structure with the following key components:

- **Encoder:**
  - Input: Extracted features from the pre-trained VGG16 model.
  - Dropout layer for regularization.
  - Dense layer with ReLU activation.
  - RepeatVector to match the sequence length.
  - Bidirectional LSTM layer for sequence processing.

- **Decoder:**
  - Input: Tokenized sequences of captions.
  - Embedding layer for word embeddings.
  - Dropout layer for regularization.
  - Bidirectional LSTM layer for sequence processing.
  - Attention mechanism using dot product.
  - Softmax activation for attention scores.
  - Lambda layer for applying attention scores to sequence embeddings.
  - Summation along the time axis to obtain context vector.
  - Concatenation of context vector and encoder output.
  - Dense layers for caption prediction.

## Training
The model is trained for a specified number of epochs using data generators for both training and validation data. The training script includes code for setting up data generators, training the model, and visualizing the model's training progress.

## Saving and Loading
Trained models can be saved to and loaded from files using the `save` and `load_model` functions from Keras.

## Image Caption Prediction
A function is provided to generate captions for new images using the trained model.

## Evaluation - BLEU Scores
BLEU-1 and BLEU-2 scores are calculated to evaluate the performance of the model on test data.

## Actual and Predicted Captions
Actual and predicted captions for test images are saved using pickle for further analysis.

## Streamlit Web Application
The `streamlit_app.py` script contains the code for deploying a user-friendly web application using Streamlit. Users can upload an image, and the application will generate and display a descriptive caption using the trained model.

To run the Streamlit app, use the following command:
```bash
streamlit run streamlit_app.py
```

Enjoy exploring the Image Caption Generator!
