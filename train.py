import keras
import keras.backend as K
import numpy as np

from keras.layers import Input, Conv3D, BatchNormalization, MaxPooling3D, UpSampling3D, Dropout, add, concatenate

from keras.models import Model, load_model, save_model
from keras.optimizers import Adam

from keras.utils import to_categorical

import nibabel as nib

import argparse


modalities = ['FLAIR', 'reg_IR', 'reg_T1']
img_shape = (240, 240, 48)

def unet(n_tissues):
    """
    3D U-net model, using very small convolutional kernels
    """

    big_conv_size = (5, 5, 5)
    small_conv_size = (3, 3, 3)
    mini_conv_size = (1, 1, 1)

    pool_size = (2, 2, 2)

    inputs = Input(shape=(img_shape[0], img_shape[1], img_shape[2], len(modalities)))

    conv1 = Conv3D(8, big_conv_size, activation='relu', padding='same', use_bias=False)(inputs)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size)(bn1)

    conv2 = Conv3D(8, big_conv_size, activation='relu', padding='same', use_bias=False)(pool1)
    conv2 = Conv3D(8, big_conv_size, activation='relu', padding='same', use_bias=False)(conv2)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size)(bn2)

    conv3 = Conv3D(8, big_conv_size, activation='relu', padding='same', use_bias=False)(pool2)
    conv3 = Conv3D(8, big_conv_size, activation='relu', padding='same', use_bias=False)(conv3)
    bn3 = BatchNormalization()(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size)(bn3)

    conv4 = Conv3D(8, big_conv_size, activation='relu', padding='same', use_bias=False)(pool3)
    conv4 = Conv3D(8, big_conv_size, activation='relu', padding='same', use_bias=False)(conv4)
    bn4 = BatchNormalization()(conv4)
    pool4 = MaxPooling3D(pool_size=pool_size)(bn4)

    # conv5 = Conv3D(32, big_conv_size, activation='relu', padding='same')(pool4)
    # bn5 = BatchNormalization()(conv5)
    # pool5 = MaxPooling3D(pool_size=pool_size)(bn5)

    conv6 = Conv3D(16, small_conv_size, activation='relu', padding='same', use_bias=False)(pool4)
    drop6 = Dropout(0.5)(conv6)
    conv7 = Conv3D(16, small_conv_size, activation='relu', padding='same', use_bias=False)(pool4)
    drop7 = Dropout(0.5)(conv7)
    conv8 = Conv3D(16, mini_conv_size, activation='relu', padding='same', use_bias=False)(pool4)
    drop8 = Dropout(0.5)(conv8)
    nadir = add([drop6, drop7, drop8])
    bn8 = BatchNormalization()(nadir)

    # skip9 = concatenate([pool5, bn8])
    # up9 = UpSampling3D(size=pool_size)(skip9)
    # conv9 = Conv3D(64, big_conv_size, activation='relu', padding='same')(up9)
    # bn9 = BatchNormalization()(conv9)

    skip10 = concatenate([pool4, bn8])
    up10 = UpSampling3D(size=pool_size)(skip10)
    conv10 = Conv3D(8, big_conv_size, activation='relu', padding='same', use_bias=False)(up10)
    conv10 = Conv3D(8, big_conv_size, activation='relu', padding='same', use_bias=False)(conv10)
    bn10 = BatchNormalization()(conv10)

    skip11 = concatenate([pool3, bn10])
    up11 = UpSampling3D(size=pool_size)(skip11)
    conv11 = Conv3D(8, big_conv_size, activation='relu', padding='same', use_bias=False)(up11)
    conv11 = Conv3D(8, big_conv_size, activation='relu', padding='same', use_bias=False)(conv11)
    bn11 = BatchNormalization()(conv11)

    skip12 = concatenate([pool2, bn11])
    up12 = UpSampling3D(size=pool_size)(skip12)
    conv12 = Conv3D(8, big_conv_size, activation='relu', padding='same', use_bias=False)(up12)
    conv12 = Conv3D(8, big_conv_size, activation='relu', padding='same', use_bias=False)(conv12)
    bn12 = BatchNormalization()(conv12)

    skip13 = concatenate([pool1, bn12])
    up13 = UpSampling3D(size=pool_size)(skip13)
    conv13 = Conv3D(16, big_conv_size, activation='relu', padding='same', use_bias=False)(up13)
    conv13 = Conv3D(16, big_conv_size, activation='relu', padding='same', use_bias=False)(conv13)
    bn13 = BatchNormalization()(conv13)

    conv14 = Conv3D(16, small_conv_size, activation='relu', padding='same', use_bias=False)(bn13)
    conv14 = Conv3D(16, small_conv_size, activation='relu', padding='same', use_bias=False)(conv14)
    drop14 = Dropout(0.1)(conv14)
    bn14 = BatchNormalization()(drop14)
    conv15 = Conv3D(8, small_conv_size, activation='relu', padding='same', use_bias=False)(bn14)
    drop15 = Dropout(0.3)(conv15)
    bn15 = BatchNormalization()(drop15)
    conv16 = Conv3D(16, mini_conv_size, activation='relu', padding='same', use_bias=False)(bn15)
    drop16 = Dropout(0.5)(conv16)
    bn17 = BatchNormalization()(drop16)

    # need as many output channel as tissue classes
    conv17 = Conv3D(n_tissues, (1, 1, 1), activation='softmax', padding='valid')(bn17)

    model = Model(inputs=[inputs], outputs=[conv17])

    return model


def batch(data_dir, subj_ids):
    img_array = np.zeros((1, 240, 240, 48, 3), dtype='float32')

    np.random.shuffle(subj_ids)
    for subj_id in subj_ids:
        img_array[0, :, :, :, 0] = nib.load(data_dir + subj_id + '/pre/FLAIR.nii.gz').get_data()
        img_array[0, :, :, :, 1] = nib.load(data_dir + subj_id + '/pre/reg_IR.nii.gz').get_data()
        img_array[0, :, :, :, 2] = nib.load(data_dir + subj_id + '/pre/reg_T1.nii.gz').get_data()

        img_array = (img_array - np.min(img_array)) / (np.max(img_array) + 0.000001)

        segmentation_img = nib.load(data_dir + subj_id + '/segm.nii.gz').get_data()

        label_array = to_categorical(segmentation_img, num_classes=11)
        label_array = np.reshape(label_array, ((1,) + img_shape + (11,)))

        yield (img_array, label_array)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a segmentation model for the MRBrains18 Challenge.')
    parser.add_argument('--data-dir', action='store', default='/data1/users/adoyle/MRBrainS18/training/', metavar='N', help='root directory for training data')
    parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train for (default: 20)')
    args = parser.parse_args()


    data_dir = args.data_dir
    n_epochs = args.epochs

    subjects = ['1', '4', '5', '7', '14', '070', '148']



    model = unet(11)

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-5)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_crossentropy', 'accuracy'])

    model.summary()

    model.fit_generator(batch(data_dir, subjects[0:-1]), len(subjects) - 1, epochs=n_epochs, validation_data=batch(data_dir, [subjects[-1]]), validation_steps=1)
