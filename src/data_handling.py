import csv
import os
import shutil

import numpy as np
import torchvision
from torch.utils.data import Subset
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split

from Global import SCREEN_WIDTH, SCREEN_HEIGHT

TEST_PERCENTAGE = 0.15
FEATURES = ['Y', 'X']
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                DEEPPROBLOG
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def get_dataset(data_dir):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    return torchvision.datasets.ImageFolder(data_dir, transform)


def get_metadata(file_path='../data/meta_data.csv'):
    metadata = list()
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append(row)
    return metadata


def split_dataset_pytorch(dataset):
    train_idx, test_idx = train_test_split(np.arange(len(dataset.targets)),
                                           test_size=TEST_PERCENTAGE,
                                           shuffle=True,
                                           stratify=dataset.targets)
    return Subset(dataset, train_idx), Subset(dataset, test_idx)


def sort_data_pytorch(metadata, unsorted_dir, sorted_dir):
    def build_sorted_dir():
        os.mkdir(sorted_dir + "/" + "up")
        os.mkdir(sorted_dir + "/" + "down")
        os.mkdir(sorted_dir + "/" + "noop")

    def sort_logic(img_data):
        if int(img_data['bally']) == int(img_data['aiy']):
            return "noop"
        elif int(img_data['bally']) > int(img_data['aiy']):
            return "down"
        return "up"

    if os.path.exists(sorted_dir):
        shutil.rmtree(sorted_dir)
    os.mkdir(sorted_dir)
    build_sorted_dir()

    for img_data in metadata:
        target = sort_logic(img_data)
        shutil.copy(os.path.join(unsorted_dir, img_data['img_id']),
                    os.path.join(sorted_dir, str(target), img_data['img_id']))


class DataSorter:
    def __init__(self):
        self.feature = None

    def sort_data_deepproblog(self, metadata, unsorted_dir, sorted_dir):
        """Sorts the images into the correct label based on the action to be taken. Removes any previously sorted_y data."""

        def build_sorted_dir_y():
            for i in range(-SCREEN_HEIGHT + 1, SCREEN_HEIGHT):
                os.mkdir(sorted_dir + "/" + str(i))

        def build_sorted_dir_x():
            for i in range(1, SCREEN_WIDTH - 1):
                os.mkdir(sorted_dir + "/" + str(i))

        def sort_logic_y(img_data):
            return int(img_data['aiy']) - int(img_data['bally'])

        def sort_logic_x(img_data):
            return int(img_data['aix']) - int(img_data['ballx'])

        # Remove previously sorted_y data
        if os.path.exists(sorted_dir):
            shutil.rmtree(sorted_dir)
        # Build sorted_y data directory tree
        os.mkdir(sorted_dir)
        build_sorted_dir_y() if self.feature == 'Y' else build_sorted_dir_x()

        # COPY all images from generated dir to sub_dir of sorted_dir based on target
        for img_data in metadata:
            target = sort_logic_y(img_data) if self.feature == 'Y' else sort_logic_x(img_data)
            shutil.copy(os.path.join(unsorted_dir, img_data['img_id']),
                        os.path.join(sorted_dir, str(target), img_data['img_id']))


    def create_query_files(self, dataset, metadata, dataset_indices, output_filename):
        def query_y(aiy, dataset_id, target):
            return f"distance_y(dataset_id({dataset_id}), {aiy}, {target}).\n"

        def query_x(dataset_id, target):
            return f"distance_x(dataset_id({dataset_id}), {target}).\n"

        examples = list()
        for i in dataset_indices:
            target = dataset.classes[dataset.targets[i]]
            if self.feature == 'Y':
                img_id = int(dataset.imgs[i][0].split('/')[-1][:-4])
                aiy = metadata[img_id]['aiy']
                examples.append(query_y(aiy=aiy, dataset_id=i, target=target))
            elif self.feature == 'X':
                examples.append(query_x(dataset_id=i, target=target))
        with open(output_filename, 'w') as f:
            for example in examples:
                f.write(example)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                    MAIN
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if __name__ == "__main__":
    """ Creates two files with respectively train- and test-querysets. This is done by first sorting the raw data
        into subdirectories of the correct label (based on which action to take) and then splitting the full dataset
        into a (stratified) train and test set.
        NOTE: the idx of an image in the dataset does not correspond to idx of the img in the metadata 
        (since the imgs are ordered by path in the dataset and by name in the metadata)
        (e.g.: @index(2) in dataset = '../data/sorted_y/down/4.png' vs @index(2) in data.csv = '2.png')."""

    metadata = get_metadata()
    data_sorter = DataSorter()
    PYTORCH_SORTED_DATA_DIR = '../data/sorted_pytorch'
    sort_data_pytorch(metadata, unsorted_dir='../data/generated', sorted_dir=PYTORCH_SORTED_DATA_DIR)

    for feature in FEATURES:
        data_sorter.feature = feature
        SORTED_DATA_DIR = '../data/sorted_y' if feature == 'Y' else '../data/sorted_x'
        TRAIN_QUERY_FILE = '../data/deepproblog_train_data_y.txt' if feature == 'Y' else '../data/deepproblog_train_data_x.txt'
        TEST_QUERY_FILE = '../data/deepproblog_test_data_y.txt' if feature == 'Y' else '../data/deepproblog_test_data_x.txt'

        data_sorter.sort_data_deepproblog(metadata, unsorted_dir='../data/generated', sorted_dir=SORTED_DATA_DIR)
        dataset = get_dataset(data_dir=SORTED_DATA_DIR)
        train_idx, test_idx = train_test_split(np.arange(len(dataset.targets)),
                                               test_size=TEST_PERCENTAGE,
                                               shuffle=True,
                                               stratify=dataset.targets)
        data_sorter.create_query_files(dataset, metadata, train_idx, TRAIN_QUERY_FILE)
        data_sorter.create_query_files(dataset, metadata, test_idx, TEST_QUERY_FILE)
