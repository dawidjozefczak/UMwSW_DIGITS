from PIL import Image, ImageOps
from numpy import asarray 
import glob
import tensorflow as tf
from tensorflow import keras
import numpy as np

path2train = 'train'
path2test = 'test'

cropX=112
cropY=208


def crop(image, left, top, right, bottom):
    cropped = image.crop((left, top, right, bottom))
    return cropped

def getFileNameNoExtension(image, path):
    if(len(path)>0):
        fileName = image.filename[len(path)+1:-4]
    else:
        fileName = image.filename[0:-4]
    return fileName

def nameCroppedImage(name, letter):
    ext = ".png"
    newName = name + letter + ext
    return newName

def openFolder(path):
    image_list = []
    for filename in glob.glob(path + '/*.png'):
        im=Image.open(filename)
        image_list.append(im)
    for filename in glob.glob(path + '/*.jpg'):
        im=Image.open(filename)
        image_list.append(im)

    return image_list

def customBinary(image, thresh):
    fn = lambda x :255 if x>thresh else 0
    r = image.convert('L').point(fn, mode='1')
    return r

def whichDigitAmI(name):
    return name[int(name[7])]

def main():
    #open images
    train_images = openFolder(path2train)
    train_labels = []
    test_images = openFolder(path2test)
    test_labels = []
    
    #create labels
    for x in range(0, len(train_images)):
        train_labels.append(whichDigitAmI(getFileNameNoExtension(train_images[x], path2train)))
    
    for x in range(0, len(test_images)):
        test_labels.append(whichDigitAmI(getFileNameNoExtension(test_images[x], path2test)))
    
    #check if any shit happened
    if(len(train_images)!=len(train_labels)):
        print("Wrong dimensions of name and image train list before training")

    if(len(test_images)!=len(test_labels)):
        print("Wrong dimensions of name and image test list before training")

    print("Size of train_images:", len(train_images))
    print("Size of test_labels:", len(test_images))

    #convert for numpy array for keras
    train_images = np.stack(train_images)
    test_images = np.stack(test_images)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    #model staff
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(cropY,cropX)),
        keras.layers.Dense(8000,activation='softsign'),
        keras.layers.Dense(1000,activation='softsign'),
        keras.layers.Dense(10,activation='softsign')
    ])

    print(model.output_shape)

    model.compile(optimizer='adadelta',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=100)

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)
    
if __name__ == '__main__':
    main()



