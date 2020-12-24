import csv
import numpy as np
import torchvision
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split


def get_dataset():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5, ))])
    return torchvision.datasets.ImageFolder('../data', transform)


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


if __name__ == "__main__":
    """ Creates a new train-test split from the images in the folder. The subdirectories are the correct labels (which
        action to take). 
        NOTE: the idx of the dataset do not correspond to the img_idx (since the imgs are ordered (1, 10,...)."""
    metadata = list()
    with open('../data/data.csv','r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append(row)

    TEST_PERCENTAGE = 0.2
    dataset = get_dataset()
    train_idx, test_idx = train_test_split(np.arange(len(dataset.targets)),
                                           test_size=TEST_PERCENTAGE,
                                           shuffle=True,
                                           stratify=dataset.targets)
    create_query_files(dataset, metadata, train_idx, 'train_data.txt')
    create_query_files(dataset, metadata, test_idx, 'test_data.txt')
