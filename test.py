import matplotlib.pyplot as plt
import os
from PIL import Image, ImageOps
from numpy import asarray 
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.saving.save import load_model

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

img_width=112
img_height=208

def main():
    num_classes = 10
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
  
    model = tf.keras.models.load_model('/home/dawid/Documents/Politechnika-mgr/Uczenie_maszynowe_w_systemach_wizyjnych/UMwSW_DIGITS/my_model.h5')
    model.summary()
    for x in range(0, 7):
        digit_path = 'check/0134152'+str(x)+'.png'

        img = keras.preprocessing.image.load_img(
              digit_path, target_size=(img_height, img_width)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )


    
if __name__ == '__main__':
    main()

