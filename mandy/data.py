"""A module containing classes and functions for data."""

import os
import pandas as pd
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision import transforms


def make_dataset(df, is_valid_file=None):
    instances = []
    paths = df['path'].values
    all_classes = df[[c for c in df.columns if c.startswith('class_')]].values
    for path, classes in zip(paths, all_classes):
        if is_valid_file is not None and not is_valid_file(path):
            continue
        item = path, classes
        instances.append(item)
    return instances


class DatasetCSV(VisionDataset):
    """A generic data loader where the samples are arranged in a csv file: ::
        path,class_x,class_y,...
        data/aaa.ext,0,0,...
        ...
        data/xxx.ext,0,1,...
        data/yyy.ext,1,0,...
        data/zzz.ext,1,1,...
    Args:
        csv_filepath (string): CSV file path.
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        tied_transform (callable, optional): A function/transform that takes in
            a sample and a target and returns a transformed version of both.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        df (pd.DataFrame): DataFrame of the csv file.
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self, csv_filepath, root, loader=default_loader, transform=None,
            target_transform=None, tied_transform=None, is_valid_file=None):
        super(DatasetCSV, self).__init__(
            root, transform=transform, target_transform=target_transform)

        df = pd.read_csv(csv_filepath)
        classes, class_to_idx = self._find_classes(df)
        samples = make_dataset(df, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.tied_transform = tied_transform
        self.loader = loader
        self.df = df
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = self.samples

    def _find_classes(self, df):
        """
        Finds the class columns in a dataset.
        Args:
            df (pd.DataFrame): data frame of the dataset.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), 
                   and class_to_idx is a dictionary.
        """
        classes = [d[6:] for d in df.columns if d.startswith('class_')]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        path = os.path.join(self.root, path)
        sample = self.loader(path)
        if self.tied_transform is not None:
            sample, target = self.tied_transform(sample, target)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, target

    def __len__(self):
        return len(self.samples)
    
    
def horizontal_flip_fx(image, target, p=0.5):
    if random.random() < p:
        image = transforms.functional.hflip(image)
        if len(target) == 9:
            for i in range(0, 7, 2):
                target[i], target[i + 1] = target[i + 1], target[i]
        elif len(target) == 3:
            y[0], y[1] = y[1], y[0]
    return image, target