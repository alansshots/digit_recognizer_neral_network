import numpy as np
import cv2 
import streamlit as st
from streamlit_drawable_canvas import st_canvas

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def get_predictions(A2):
    return np.argmax(A2, 0)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

# Load trained parameters
def load_params(directory):
    W1 = np.loadtxt(f'{directory}/W1.csv', delimiter=',')
    b1 = np.loadtxt(f'{directory}/b1.csv', delimiter=',').reshape(-1, 1)
    W2 = np.loadtxt(f'{directory}/W2.csv', delimiter=',')
    b2 = np.loadtxt(f'{directory}/b2.csv', delimiter=',').reshape(-1, 1)
    return W1, b1, W2, b2

W1, b1, W2, b2 = load_params('./neural_network/trained_data') 

# Streamlit app
st.title('Hand-Drawn Digit Recognizer Using MNIST Dataset')
st.markdown('Draw a digit and click "predict" to see result.')

SIZE = 300

col1, col2 = st.columns(2)

with col1:
    canvas_result = st_canvas(
        fill_color='#000000',
        stroke_width=20,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=SIZE,
        height=SIZE,
        drawing_mode="freedraw",
        key='canvas'
    )

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)  
    with col2:
        st.write('Model Input')
        st.image(rescaled)

if st.button('Predict'):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = img.reshape(784, 1)
    prediction = make_predictions(img, W1, b1, W2, b2)
    st.write(f'Prediction: {prediction}')
