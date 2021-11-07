from PIL import Image, ImageOps
from numpy import asarray 
import glob
import tensorflow as tf
from tensorflow import keras
import numpy as np

path2crop = 'before_processing'
cropped_path = 'croped/'

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

    #open images from directory
    image_list = openFolder(path2crop)
    
    #crop chosen images to some small parts contained only digits
    cropped_list = []
    
    for x in image_list:
        cropped_list.append(crop(x,1367, 831, 1367+cropX, 831+cropY))
        cropped_list.append(crop(x,1601, 831, 1601+cropX, 831+cropY))
        cropped_list.append(crop(x,1829, 841, 1829+cropX, 841+cropY))
        cropped_list.append(crop(x,2049, 845, 2049+cropX, 845+cropY))
        cropped_list.append(crop(x,2287, 845, 2287+cropX, 845+cropY))
        cropped_list.append(crop(x,2507, 845, 2507+cropX, 845+cropY))
        cropped_list.append(crop(x,2715, 845, 2715+cropX, 845+cropY))
           
    #convert image to greyscale
    grey_list = []

    for filename in cropped_list:
        grey=filename.convert('L')
        grey_list.append(grey)

    #invert colors in image
    inv_list = []

    for filename in grey_list:
        inv = ImageOps.invert(filename)
        inv_list.append(inv)

    #set threshold to convert image to binary
    bw_list = []

    for filename in inv_list:
        bw = customBinary(filename, 215)
        bw_list.append(bw)

    #save cropped and modified images to chosen directory
    for i in range(0, len(image_list)):
        for j in range(0,7):
            bw_list[j+i*7].save(cropped_path + nameCroppedImage(getFileNameNoExtension(image_list[i], path2crop), str(j)))
    
if __name__ == '__main__':
    main()



