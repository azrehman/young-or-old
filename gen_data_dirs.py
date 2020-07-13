import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm


# TODO shuffle images
def gen_data_dirs(image_dir, class_file, TRAIN_RATIO=0.8):
    print('moving and sorting data....')
    project_dir = os.getcwd()

    # directories to create
    train_path_young = 'data/train/young'
    train_path_old = 'data/train/old'

    val_path_young = 'data/validation/young'
    val_path_old = 'data/validation/old'

    # load attributes using pandas
    attributes = pd.read_csv(class_file)

    # read attributes class to sort data
    y = attributes.Young
    y = y.replace(to_replace=-1, value=0)
    y = y.to_numpy()

    # create list of image names
    image_names = attributes.image_id
    image_names = image_names.to_numpy()

    # change to project dir and create dirs
    os.chdir(project_dir)
    os.makedirs(train_path_young)
    os.makedirs(train_path_old)
    os.makedirs(val_path_young)
    os.makedirs(val_path_old)

    # calculate training/val proportion
    n_images = len(y)
    n_training = int(TRAIN_RATIO * n_images)

    for i, image_name in enumerate(tqdm(image_names)):
        # index image using filename with extension removed
        # subtract 1 because image names start at 1 not 0
        index = int(image_name.split('.')[0]) - 1
        # add images to training dir
        if i < n_training:
            # old subdir
            if y[index] == 0:
                shutil.copy(os.path.join(image_dir, image_name),
                            train_path_old)
            # young subdir
            elif y[index] == 1:
                shutil.copy(os.path.join(image_dir, image_name),
                            train_path_young)
        # add images to validation dir
        else:
            # old subdir
            if y[index] == 0:
                shutil.copy(os.path.join(image_dir, image_name),
                            val_path_old)
            # young subdir
            elif y[index] == 1:
                shutil.copy(os.path.join(image_dir, image_name),
                            val_path_young)
    print('done')


# locations of face images folder and attributes csv file
IMAGES_PATH = 'C:\\Datasets\\celebA\\img_align_celeba\\img_align_celeba'
# location of csv file
ATTRIBUTE_PATH = 'C:\\Datasets\\celebA\\list_attr_celeba.csv'
# generate training, validation directories
# from gen_data_dirs import gen_data_dirs
gen_data_dirs(IMAGES_PATH, ATTRIBUTE_PATH, TRAIN_RATIO=0.8)
