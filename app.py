import os
import sys

# Set encoding to UTF-8 for stdout
sys.stdout.reconfigure(encoding='utf-8')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO messages

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model , Model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation


app = Flask(__name__)

# ******************************** classification  ********************************
# Load the classification model
model_path = "classification_model.keras"
model = load_model(model_path)

class_mappings = {'Glioma': 0, 'Meninigioma': 1, 'Notumor': 2, 'Pituitary': 3}

# Function to preprocess the image for classification
def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('L')  # Open image in grayscale mode
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    return img_array

def classify_image(image_path):
    # Preprocess the uploaded image
    img = preprocess_image(image_path)
    # Perform classification
    prediction = model.predict(img)
    # Get the predicted class
    predicted_class_index = np.argmax(prediction)
    predicted_class = list(class_mappings.keys())[predicted_class_index]
    return predicted_class

# ******************************** segmentation ********************************
def conv_block1(inputs, filters):
    x = Conv2D(filters, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def encoder_block1(inputs, filters):
    x = conv_block1(inputs, filters)
    p = MaxPooling2D(pool_size=(2, 2))(x)
    return x, p

def decoder_block1(inputs, filters, concat_layer):
    x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(inputs)
    x = concatenate([x, concat_layer])
    x = conv_block1(x, filters)
    return x

def unet(input_shape):
    inputs = Input(input_shape)
    s1, p1 = encoder_block1(inputs, 32) 
    s2, p2 = encoder_block1(p1, 64)
    s3, p3 = encoder_block1(p2, 128)
    s4, p4 = encoder_block1(p3, 256)
    b1 = conv_block1(p4, 512) 
    d1 = decoder_block1(b1, 256, s4)  
    d2 = decoder_block1(d1, 128, s3)
    d3 = decoder_block1(d2, 64, s2)
    d4 = decoder_block1(d3, 32, s1)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(d4)

    unet_model = Model(inputs, outputs, name="UNet")
    return unet_model

unet_model = unet((256, 256, 1))

# Load the model
model_path = 'segmentation_model.hdf5'
unet_model.load_weights(model_path)

def highlight_tumor(image_path):
    save_path = 'static/uploads/highlighted_img.jpg' 
    # Read the image
    image = Image.open(image_path).convert('L')  # Open image in grayscale mode
    image = image.resize((256, 256))
    image = np.array(image) / 255.0  # Normalize the image
    # Expand dimensions to match unet_model input shape
    image = np.expand_dims(image, axis=(0, -1))
    # Perform segmentation
    mask = unet_model.predict(image)
    # Threshold the mask
    threshold = 0.5
    mask_binary = (mask > threshold).astype(np.uint8)
    # Convert the original image to RGB (for visualization purposes)
    original_image_rgb = Image.open(image_path).convert('RGB')
    original_image_rgb = np.array(original_image_rgb)
    # Resize the mask to match the size of the original image
    highlight_mask_resized = np.array(Image.fromarray(mask_binary[0, :, :, 0]).resize((original_image_rgb.shape[1], original_image_rgb.shape[0])))
    # Create a mask to highlight the tumor region in red
    highlight_mask = np.zeros_like(original_image_rgb)
    highlight_mask[:, :, 0] = np.where(highlight_mask_resized > 0, 255, 0)  # Set red channel to 255 where tumor is present
    # Combine the original image with the highlight mask
    highlighted_image = np.clip(0.7 * original_image_rgb + 0.3 * highlight_mask, 0, 255).astype(np.uint8)
    # Save the highlighted image
    Image.fromarray(highlighted_image).save(save_path)


# ******************************** Application ********************************
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the uploaded file
        imagefile = request.files['imagefile']

        # Check if the file is selected
        if imagefile.filename == '':
            return render_template('index.html', error="No file selected. Please select an image file."), 400

        # Save the file
        image_path = "static/uploads/original.jpg"
        imagefile.save(image_path)

        predicted_class = classify_image(image_path)

        if predicted_class != "Notumor":
            highlight_tumor(image_path) 

        return render_template('result.html', predicted_class=predicted_class)

    return render_template('index.html')

@app.route('/admin', methods=['GET', 'POST'])
def login():
    return render_template('login.html')

@app.route('/messages', methods=['GET', 'POST'])
def messages():
    return render_template('messages.html')


if __name__ == '__main__':
    app.run(debug=False)