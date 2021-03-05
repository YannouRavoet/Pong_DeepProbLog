import csv
import os
import shutil

import numpy as np
import torchvision
from torch.utils.data import Subset
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split

TEST_PERCENTAGE = 0.2
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                DEEPPROBLOG
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def get_metadata(file_path='../data/data.csv'):
    metadata = list()
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append(row)
    return metadata


def sort_data_deepproblog(metadata, unsorted_dir='../data/generated', sorted_dir='../data/sorted_deepproblog'):
    """Sorts the images into the correct label based on the action to be taken. Removes any previously sorted_deepproblog data."""

    def action_logic(img_data):
        if int(img_data['bally']) == int(img_data['aiy']):
            return "noop"
        elif int(img_data['bally']) > int(img_data['aiy']):
            return "down"
        return "up"

    # Remove previously sorted_deepproblog data
    if os.path.exists(sorted_dir):
        shutil.rmtree(sorted_dir)
    # Build sorted_deepproblog data directory tree
    os.mkdir(sorted_dir)
    os.mkdir(sorted_dir + "/" + "up")
    os.mkdir(sorted_dir + "/" + "down")
    os.mkdir(sorted_dir + "/" + "noop")

    # COPY all images from generated dir to sub_dir of sorted_dir based on target
    for img_data in metadata:
        desired_action = action_logic(img_data)
        shutil.copy(os.path.join(unsorted_dir, img_data['img_id']),
                    os.path.join(sorted_dir, desired_action, img_data['img_id']))


def get_dataset(data_dir='../data/sorted_deepproblog'):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    return torchvision.datasets.ImageFolder(data_dir, transform)


def next_example(aiy, dataset_id, target):
    return f"choose_action({aiy}, dataset_id({dataset_id}), {target}).\n"


def create_query_files(dataset, metadata, dataset_indices, output_filename):
    examples = list()
    for i in dataset_indices:
        target = dataset.classes[dataset.targets[i]]
        img_id = int(dataset.imgs[i][0].split('/')[-1][:-4])
        examples.append(next_example(aiy=metadata[img_id]['aiy'],
                                     dataset_id=i,
                                     target=target))
    with open(output_filename, 'w') as f:
        for example in examples:
            f.write(example)


def split_dataset_pytorch(dataset):
    train_idx, test_idx = train_test_split(np.arange(len(dataset.targets)),
                                           test_size=TEST_PERCENTAGE,
                                           shuffle=True,
                                           stratify=dataset.targets)
    return Subset(dataset, train_idx), Subset(dataset, test_idx)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                    MAIN
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if __name__ == "__main__":
    """ Creates two files with respectively train- and test-querysets. This is done by first sorting the raw data
        into subdirectories of the correct label (based on which action to take) and then splitting the full dataset
        into a (stratified) train and test set.
        NOTE: the idx of an image in the dataset does not correspond to idx of the img in the metadata 
        (since the imgs are ordered by path in the dataset and by name in the metadata)
        (e.g.: '../data/sorted_deepproblog/down/2.png' vs '2.png')."""
    metadata = get_metadata()
    sort_data_deepproblog(metadata)
    dataset = get_dataset()
    train_idx, test_idx = train_test_split(np.arange(len(dataset.targets)),
                                           test_size=TEST_PERCENTAGE,
                                           shuffle=True,
                                           stratify=dataset.targets)
    create_query_files(dataset, metadata, train_idx, '../data/deepproblog_train_data.txt')
    create_query_files(dataset, metadata, test_idx, '../data/deepproblog_test_data.txt')
