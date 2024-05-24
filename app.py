from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
import numpy as np
import tensorflow as tf
from pyngrok import ngrok

app = Flask(__name__)

# Load your trained model
model_path = '/content/drive/MyDrive/model_cnn (1).h5'
model = tf.keras.models.load_model(model_path)


def preprocess_image(image):
    image = image.resize((224, 224))  # resize to your model's input shape
    image = np.array(image)
    image = image / 255.0  # normalize
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image']
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)
    result = np.argmax(prediction, axis=1)

    return jsonify({'prediction': int(result[0])})

# Set up Ngrok with authtoken
ngrok.set_auth_token("2gt37omtrSZqxxeoZvXrdhsB5O4_7naBY6aXXGaPCmruxrMwr")

# Expose the Flask app to the web
public_url = ngrok.connect(5000)
print('Public URL:', public_url)
app.run(port=5000)
