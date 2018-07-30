from keras.models import load_model

import argparse
import nibabel as nib
import numpy as np


def predict_segmentation(input_images_path):
    img_array = np.zeros((1, 240, 240, 48, 3), dtype='float32')

    img_array[0, :, :, :, 0] = nib.load(input_images_path + 'FLAIR.nii.gz').get_data()
    img_array[0, :, :, :, 1] = nib.load(input_images_path + 'reg_IR.nii.gz').get_data()
    img_array[0, :, :, :, 2] = nib.load(input_images_path + 'reg_T1.nii.gz').get_data()

    prediction_one_hot = model.predict(img_array)

    prediction = np.argmax(prediction_one_hot, axis=-1)



    return prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a segmentation model for the MRBrains18 Challenge.')
    parser.add_argument('--data-dir', action='store', default='/data1/users/adoyle/MRBrainS18/training/', metavar='N', help='root directory for training data')
    args = parser.parse_args()

    data_dir = args.data_dir

    subjects = ['1', '4', '5', '7', '14', '070', '148']

    header = nib.load(data_dir + '1/segm.nii.gz').header

    model = load_model(data_dir + 'neuroMTL_segmentation.hdf5')

    for subj_id in subjects:
        predicted_img = predict_segmentation(data_dir + subj_id + '/pre/')
        save_img = nib.Nifti2Image(predicted_img, np.eye(4))
        save_img.header = header
        nib.save(save_img, data_dir + subj_id + '_segmented.nii.gz')