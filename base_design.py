import numpy as np
import math
import argparse
import os
import gc
from PIL import Image
import time
import tensorflow as tf
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation, Average
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.core import Reshape
from keras.layers import Embedding
from keras.layers import concatenate
import itertools as it
import random
import scipy.io as sio
gc.enable() 


def create_cnn(width, height, depth): #CNN design modelled after googlenet inception based modules
	# initialize the input shape and channel dimension, assuming
    # The width and height of the input images in pixels
    # define the model input
    inputShape = (width, height, depth)
    input_image = Input(shape=inputShape, name='input_image')
    #each block represents a single inception module 
    conv1 = Conv2D(16, kernel_size=(3, 3), strides=1, data_format="channels_last", padding='same', activation='relu')(input_image)
    conv1 = Conv2D(32, kernel_size=(3, 3), strides=1, data_format="channels_last", padding='same', activation='relu')(conv1)
    # Start of first inception module 
    intconv1_1 = Conv2D(64, kernel_size=(1, 1), strides=1, data_format="channels_last", padding='same', activation='relu')(conv1)    
    intconv1_1 = Conv2D(16, kernel_size=(1, 1), strides=1, data_format="channels_last", padding='same', activation='linear')(intconv1_1)

    intconv1_2 = Conv2D(64, kernel_size=(3, 3), strides=1, data_format="channels_last", padding='same', activation='relu')(conv1)
    intconv1_2 = Conv2D(16, kernel_size=(1, 1), strides=1, data_format="channels_last", padding='same', activation='linear')(intconv1_2)

    intconv1_3 = Conv2D(64, kernel_size=(5, 5), strides=1, data_format="channels_last", padding='same', activation='relu')(conv1)
    intconv1_3 = Conv2D(16, kernel_size=(1, 1), strides=1, data_format="channels_last", padding='same', activation='linear')(intconv1_3)

    intconv1_4 = MaxPooling2D(pool_size=(3, 3), strides=1, data_format="channels_last", padding='same')(conv1)

    intconv1 = concatenate([intconv1_1, intconv1_2, intconv1_3, intconv1_4])  #end of first inception module


    conv2 = Conv2D(32, kernel_size=(3, 3), strides=1, data_format="channels_last", padding='same', activation='relu')(intconv1)    
    conv2 = MaxPooling2D(pool_size=(3, 3), strides=2, data_format="channels_last", padding='same')(conv2)
    # Start of Second inception module 
    intconv2_1 = Conv2D(64, kernel_size=(1, 1), strides=1, data_format="channels_last", padding='same', activation='relu')(conv2)
    intconv2_1 = Conv2D(16, kernel_size=(1, 1), strides=1, data_format="channels_last", padding='same', activation='linear')(intconv2_1)

    intconv2_2 = Conv2D(64, kernel_size=(3, 3), strides=1, data_format="channels_last", padding='same', activation='relu')(conv2)
    intconv2_2 = Conv2D(16, kernel_size=(1, 1), strides=1, data_format="channels_last", padding='same', activation='linear')(intconv2_2)

    intconv2_3 = Conv2D(64, kernel_size=(5, 5), strides=1, data_format="channels_last", padding='same', activation='relu')(conv2)
    intconv2_3 = Conv2D(16, kernel_size=(1, 1), strides=1, data_format="channels_last", padding='same', activation='linear')(intconv2_3)

    intconv2_4 = MaxPooling2D(pool_size=(3, 3), strides=1, data_format="channels_last", padding='same')(conv2)

    intconv2 = concatenate([intconv2_1, intconv2_2, intconv2_3, intconv2_4])     # End of second inception module 


    conv3 = Conv2D(32, kernel_size=(3, 3), strides=1, data_format="channels_last", padding='same', activation='relu')(intconv2)    
    conv3 = MaxPooling2D(pool_size=(3, 3), strides=2, data_format="channels_last", padding='same')(conv3)
    # Start of third inception module 
    intconv3_1 = Conv2D(64, kernel_size=(1, 1), strides=1, data_format="channels_last", padding='same', activation='relu')(conv3)
    intconv3_1 = Conv2D(16, kernel_size=(1, 1), strides=1, data_format="channels_last", padding='same', activation='linear')(intconv3_1)

    intconv3_2 = Conv2D(64, kernel_size=(3, 3), strides=1, data_format="channels_last", padding='same', activation='relu')(conv3)
    intconv3_2 = Conv2D(16, kernel_size=(1, 1), strides=1, data_format="channels_last", padding='same', activation='linear')(intconv3_2)

    intconv3_3 = Conv2D(64, kernel_size=(5, 5), strides=1, data_format="channels_last", padding='same', activation='relu')(conv3)
    intconv3_3 = Conv2D(16, kernel_size=(1, 1), strides=1, data_format="channels_last", padding='same', activation='linear')(intconv3_3)

    intconv3_4 = MaxPooling2D(pool_size=(3, 3), strides=1, data_format="channels_last", padding='same')(conv3)

    intconv3 = concatenate([intconv3_1, intconv3_2, intconv3_3, intconv3_4])     # End of Third inception module 


    conv4 = Conv2D(32, kernel_size=(3, 3), strides=1, data_format="channels_last", padding='same', activation='relu')(intconv3)    
    conv4 = MaxPooling2D(pool_size=(3, 3), strides=2, data_format="channels_last", padding='same')(conv4)
    # Start of Fourth inception module 
    intconv4_1 = Conv2D(64, kernel_size=(1, 1), strides=1, data_format="channels_last", padding='same', activation='relu')(conv4)
    intconv4_1 = Conv2D(16, kernel_size=(1, 1), strides=1, data_format="channels_last", padding='same', activation='linear')(intconv4_1)

    intconv4_2 = Conv2D(64, kernel_size=(3, 3), strides=1, data_format="channels_last", padding='same', activation='relu')(conv4)
    intconv4_2 = Conv2D(16, kernel_size=(1, 1), strides=1, data_format="channels_last", padding='same', activation='linear')(intconv4_2)

    intconv4_3 = Conv2D(64, kernel_size=(5, 5), strides=1, data_format="channels_last", padding='same', activation='relu')(conv4)
    intconv4_3 = Conv2D(16, kernel_size=(1, 1), strides=1, data_format="channels_last", padding='same', activation='linear')(intconv4_3)

    intconv4_4 = MaxPooling2D(pool_size=(3, 3), strides=1, data_format="channels_last", padding='same')(conv4)

    intconv4 = concatenate([intconv4_1, intconv4_2, intconv4_3, intconv4_4])     # Start of fourth inception module 


    conv5 = Conv2D(32, kernel_size=(3, 3), strides=1, data_format="channels_last", padding='same', activation='relu')(intconv4)    
    conv5 = MaxPooling2D(pool_size=(3, 3), strides=2, data_format="channels_last", padding='same')(conv5)
    # Start of fifth inception module 
    intconv5_1 = Conv2D(64, kernel_size=(1, 1), strides=1, data_format="channels_last", padding='same', activation='relu')(conv5)
    intconv5_1 = Conv2D(16, kernel_size=(1, 1), strides=1, data_format="channels_last", padding='same', activation='linear')(intconv5_1)

    intconv5_2 = Conv2D(64, kernel_size=(3, 3), strides=1, data_format="channels_last", padding='same', activation='relu')(conv5)
    intconv5_2 = Conv2D(16, kernel_size=(1, 1), strides=1, data_format="channels_last", padding='same', activation='linear')(intconv5_2)

    intconv5_3 = Conv2D(64, kernel_size=(5, 5), strides=1, data_format="channels_last", padding='same', activation='relu')(conv5)
    intconv5_3 = Conv2D(16, kernel_size=(1, 1), strides=1, data_format="channels_last", padding='same', activation='linear')(intconv5_3)

    intconv5_4 = MaxPooling2D(pool_size=(3, 3), strides=1, data_format="channels_last", padding='same')(conv5)

    intconv5 = concatenate([intconv5_1, intconv5_2, intconv5_3, intconv5_4])    # End of fifth inception module 

    conv6 = Conv2D(32, kernel_size=(3, 3), strides=1, data_format="channels_last", padding='same', activation='relu')(intconv5)    

    flat = Flatten()(conv6) # flatten conv layers into dense layer

    dense = Dense(512, activation='relu')(flat)
    main_output = Dense(196, activation='softmax')(dense) #196 outputs as dataset has 196 classes
    model = Model(inputs=[input_image], outputs=[main_output])
    return model 


def save_image(filename, norm_arr): # function for saving image, used for debugging and testing only
    img = (norm_arr.copy()).astype(np.uint8)

    res = Image.fromarray(img).convert('L')
    res.save(filename)

def load_data(folder_path, mat_name): #function for loading test and training data
    os.chdir(folder_path)
    mat = sio.loadmat(str(mat_name) +".mat", squeeze_me=True)
    os.chdir('..')
    idx_name = mat['annotations']
    return idx_name

def load_meta(folder_path, mat_name): #function for loading test and metadata
    os.chdir(folder_path)
    mat = sio.loadmat(str(mat_name) +".mat", squeeze_me=True)
    os.chdir('..')
    idx_name = mat['class_names']
    return idx_name

def load_imgs(folder_path, img_name, width, height): #function for loading and resizing images
    os.chdir(folder_path)
    img = Image.open(str(img_name))
    os.chdir('..')
    return img

def filter_dataset( dataset): #filtering dataset for testing data
    filtering = np.zeros( (len(dataset) ), dtype=np.float)
    for idx in range(len(dataset)): 
        test_flag  = dataset[idx][6]
        if test_flag == 1 :
            filtering[idx] = True
        else:
            filtering[idx] = False
    return filtering

def jpg_crop_grey_arr(image_name, width, height): #function for loading and resizing images
    with Image.open(str(image_name)) as img:
        lower_x = train_info[0][0]
        upper_x = train_info[0][2]
        lower_y = train_info[0][1]
        upper_y = train_info[0][3]
        bound_x = upper_x - lower_x 
        bound_y = upper_y - lower_y
        img = np.array(img)
        if img.ndim < 3: #for normalizing images that only have one channel
            img_new = np.zeros( (np.size(img,0),np.size(img,1),3), dtype=np.float)
            img_new[:,:,0] = img [:,:]
            img_new[:,:,1] = img [:,:]
            img_new[:,:,2] = img [:,:]
            img = img_new
        else :
            pass

        if upper_y >= np.size(img, 0) : #for filtering images with incorrect boundary conditions
            img_crop = img
        elif upper_x >= np.size(img, 1) :
            img_crop = img
        else :    
            img_crop = np.zeros( (bound_y,bound_x,3), dtype=np.float)
            for y_pos in range (bound_y): 
                for x_pos in range (bound_x):
                    img_crop[y_pos, x_pos, 0:2] = img[ lower_y + y_pos , lower_x + x_pos, 0:2]
        img_crop = (img_crop.copy()).astype(np.uint8)
        img_crop = Image.fromarray(img_crop)
        img_crop = img_crop.resize((width,height), Image.ANTIALIAS) #resize to standard resolution
        img_crop = np.array(img_crop)
    return img_crop

def aug_data(raw_arr): # augmentation of training data to improve training performance 
    res_LR = np.flip(raw_arr, 1) #flips left right 

    res_90 = np.rot90(raw_arr, 1) #90 rotation
    res_90LR = np.flip(res_90, 1) #90 + left right flip
    
    res_180 = np.rot90(raw_arr, 2) #180 rotation 
    res_LR180 = np.flip(res_180, 1) #180 + left right flip 
    
    res_270 = np.rot90(raw_arr, 3) #90 rotation
    res_270LR = np.flip(res_270, 1) #90 + left right flip    
    res = np.stack((raw_arr, res_90, res_90LR, res_180,res_LR,res_LR180, res_270, res_270LR), axis = 0 )
    return res

def data_load(folder_path, width, height): # function for loading images 
    os.chdir(folder_path)
    img_arr = jpg_crop_grey_arr(train_info[0][5], width, height)      # apply greyscale and cropping
    img_arr_stack = np.reshape(img_arr, (1,128,128,3))
    for file_idx in range (2,len(train_info)):
        img_arr = jpg_crop_grey_arr(train_info[file_idx][5], width, height)
        img_arr = np.reshape(img_arr, (1,128,128,3))
        img_arr_stack = np.concatenate((img_arr_stack,img_arr), axis = 0 )
    os.chdir('..')
    return img_arr

def class_load(dataset): # function for loading class dataset
    train_class = np.zeros((len(dataset)), dtype=np.float)
    for file_idx in range (len(dataset)):
        train_class[file_idx] = dataset[file_idx][4]
    return train_class

def test_data_process(folder_path, batch_size, width, height, filtering): # function to load test and train images and classes
    os.chdir(folder_path) 
    test_class = np.zeros((8041), dtype=np.int) #number of testing images specified 
    train_class = np.zeros((8144), dtype=np.int) #number of training images specified 
    test_class_count = 0 
    train_class_count = 0
    for file_idx in range (len(test_info)):
        if filtering[file_idx] == True:
            test_class[test_class_count] = test_info[file_idx][5]
            test_class_count = test_class_count +1
        else:
            train_class[train_class_count] = test_info[file_idx][5]
            train_class_count = train_class_count +1
    test_img_stack = np.zeros( (8041,128,128,3) , dtype=np.float) # creating and sorting test and training class datasets
    train_img_stack = np.zeros((8144,128,128,3), dtype=np.float)
    train_count = 0
    test_count = 0 
    for file_idx in range (len(test_info)):
        if filtering[file_idx] == True:
            text = test_info[file_idx][0][:0] + test_info[file_idx][0][8:] 
            test_img_stack[test_count, :, :, :] = jpg_crop_grey_arr(text, width, height)
            test_count = test_count + 1
        else : 
            text = test_info[file_idx][0][:0] + test_info[file_idx][0][8:] 
            train_img_stack[train_count,:,:,:] = jpg_crop_grey_arr(text, width, height)
            train_count = train_count + 1
    os.chdir('..')
    return train_img_stack, test_img_stack, train_class, test_class

def hot_encode(dataset, num_classes): #function for hot encoding the dataset the sparse categorical format for softmax activation 
    data_classes = np.zeros( (len(dataset), num_classes) , dtype=np.int)
    for idx in range(len(dataset)): 
        data_classes[idx, dataset[idx]-1 ] = 1
    return data_classes 

def data_gen(dataset, classes, num_classes): #generator for network input #generator improves efficency and RAM management since GPU training is used 
    while True: 
        class_set = np.zeros((8, num_classes), dtype=np.int)
        for idx in range(len(dataset)): 
            img_arr = dataset[idx, : , : , :]
            img_arr = aug_data(img_arr)
            for class_idx in range(8) : 
                class_set[class_idx, :] = classes[idx, :]
            yield img_arr, class_set

def data_reduction(dataset): #reduces images to one channel 
    reduc_dataset = np.zeros((len(dataset),128,128,1), dtype=np.float)
    reduc_dataset[:,:,:,0] = dataset [:,:,:, 0] 
    return reduc_dataset

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--filepath', required=True, help="specify filepath")
    args = ap.parse_args()
    os.chdir(args.filepath)
    #create network    
    model = create_cnn(128, 128, 1)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    #loading training data
    train_info = load_data('car_devkit', 'cars_train_annos') 
    print(np.array(train_info).shape)
    print(len(train_info[1] ))

    #loading testing data
    test_info = load_data('car_devkit', 'cars_annos')
    filtering = filter_dataset(test_info)
    print(len(test_info))
    print(len(filtering))

    #loading meta data
    meta_info = load_meta('car_devkit', 'cars_meta') 
    print(len(meta_info))

    #loading and processing training and testing image dataset and classes
    print('dataset 1 processing')
    (train_dataset, test_dataset, train_classes, test_classes)= test_data_process('car_ims', 4, 128, 128, filtering)
    print(np.array(train_dataset).shape)
    print(np.array(test_dataset).shape)
    print(np.array(train_classes).shape)
    print(np.array(test_classes).shape)

    train_classes = hot_encode(train_classes, len(meta_info)) #hot encode class dataset for network training
    print(np.array(train_classes).shape)  
    test_classes = hot_encode(test_classes,len(meta_info))
    print(np.array(test_classes).shape)

    test_dataset = data_reduction(test_dataset)
    train_dataset = data_reduction(train_dataset)
    print('final')
    print(np.array(train_dataset).shape)
    print(np.array(test_dataset).shape)

    # save_image(str('test')+".jpg", new_img)

    gc.collect()
    #training network
    model.fit_generator(data_gen(train_dataset,train_classes,len(meta_info)), steps_per_epoch=len(train_classes), epochs=100, validation_data=(test_dataset[:1000], test_classes[:1000]), max_queue_size=100, workers=1, use_multiprocessing=False, verbose = 1)
    gc.collect()
    #testing network on evaluation dataset, img number 1001 to 8041 of the testing dataset
    score = model.evaluate(test_dataset[1001:], test_classes[1001:], verbose=1)
    print('\n', 'Test accuracy:', score[1])
