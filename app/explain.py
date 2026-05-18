import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from preprocess import preprocess_image_input

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR,
    "..",
    "model",
    "densenet_freeze.h5"
)

model= load_model(model_path)
last_conv_layer= "conv5_block16_concat"

def generate_gradcam(input_image):
    img_array= preprocess_image_input(input_image)
    
    grad_model= tf.keras.models.Model(
        [model.inputs],
        [
            model.get_layer(last_conv_layer).output,
            model.output
        ]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]
    
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (224, 224))
    
    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    original_img = cv2.imread(input_image)
    original_img = cv2.resize(original_img, (224, 224))

    superimposed_img = cv2.addWeighted(
        original_img,
        0.6,
        heatmap,
        0.4,
        0
    )

    return superimposed_img