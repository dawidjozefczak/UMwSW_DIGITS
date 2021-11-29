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
import pandas as pd

img_width=112
img_height=208
batch_size = 32

#change in your OS - main set of images
data_dir = "/home/dawid/Documents/Politechnika-mgr/Uczenie_maszynowe_w_systemach_wizyjnych/UMwSW_DIGITS/classes"

# def crop(image, left, top, right, bottom):
#     cropped = image.crop((left, top, right, bottom))
#     return cropped

# def getFileNameNoExtension(image, path):
#     if(len(path)>0):
#         fileName = image.filename[len(path)+1:-4]
#     else:
#         fileName = image.filename[0:-4]
#     return fileName

# def nameCroppedImage(name, letter):
#     ext = ".png"
#     newName = name + letter + ext
#     return newName

# def openFolder(path):
#     image_list = []
#     for filename in glob.glob(path + '/*.png'):
#         im=Image.open(filename)
#         image_list.append(im)
#     for filename in glob.glob(path + '/*.jpg'):
#         im=Image.open(filename)
#         image_list.append(im)

#     return image_list

# def customBinary(image, thresh):
#     fn = lambda x :255 if x>thresh else 0
#     r = image.convert('L').point(fn, mode='1')
#     return r

# def whichDigitAmI(name):
#     return name[int(name[7])]

def main():
    
    #train and validation sets
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)

    #network project and settings
    num_classes = 10

    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
    #           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #           metrics=['accuracy'])

    model.summary()
    img_file = './model_arch.png'
    tf.keras.utils.plot_model(model, to_file=img_file, show_shapes=True, show_layer_names=True)
   
    epochs=10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        use_multiprocessing=True
    )
   
   
    model.save("my_model.h5")

    #result characteristics
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.title('Training and Validation')
    plt.show()


    # #extra test - group of images outside main image set
    
    # for x in range(0, 7):
    #     digit_path = 'check/0134152'+str(x)+'.png'

    #     img = keras.preprocessing.image.load_img(
    #         digit_path, target_size=(img_height, img_width)
    #     )
    #     img_array = keras.preprocessing.image.img_to_array(img)
    #     img_array = tf.expand_dims(img_array, 0) # Create a batch

    #     predictions = model.predict(img_array)
    #     score = tf.nn.softmax(predictions[0])

    #     print(
    #         "This image most likely belongs to {} with a {:.2f} percent confidence."
    #         .format(class_names[np.argmax(score)], 100 * np.max(score))
    #     )
    
if __name__ == '__main__':
    main()



