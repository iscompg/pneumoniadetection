import tensorflow as tf
import numpy as np


from tensorflow.keras.applications.densenet import preprocess_input


image_size_input=(224,224)

def preprocess_image_input(input_image):
    img=tf.keras.preprocessing.image.load_img(
        input_image,
        target_size=image_size_input
    )
    
    img_array= tf.keras.preprocessing.image.img_to_array(img)
    img_array= np.expand_dims(img_array, axis=0)
    
    img_array=preprocess_input(img_array)
    
    return img_array
    