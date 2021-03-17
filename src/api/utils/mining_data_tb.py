import tensorflow as tf
import os
import numpy as np
import cv2 as cv2 

#give format to the augmentator

data_augmentation = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.1,
    channel_shift_range=0.0,
    fill_mode="nearest",
    cval=0.0,
    horizontal_flip=True,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=None,
    validation_split=0.0,
    dtype=None,
)
cwd = os.getcwd()
def augmentator(num_samples):
    ACUM = 0
    for i in data_augmentation.flow_from_directory(cwd + "\\Neural_project\\src\\api\\",
                                                classes=["uploads"], batch_size = 1, save_to_dir=cwd + '\\Neural_project\\src\\api\\uploads\\new',
                                                save_prefix='test_vir_aug',
                                                save_format='jpeg'):
        ACUM += 1
        if ACUM == num_samples:
            break
    print("Created " + str(num_samples) + " new photos for the dataset")


def cargar_imagenes(folder):
    images = []
    print("Reading Files....")
    for filename in os.listdir(folder):
        #print("this his the filename" + filename)
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_NEAREST)
        if img is not None:
            images.append(img)      
    return np.asarray(images)      

