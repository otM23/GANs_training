# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 22:18:40 2021

@author: othmane.mounjid
"""

# import libraries
import os
import numpy as np
import torch.utils.data as data
import pickle

# bool that generate data when true
gen_data = False

# generate data
if gen_data:
    mean = 3
    var = 1
    in_shape = 1
    data_size_train = 50000
    data_size_test = 1000
    
    root = '../../data'
    root = os.path.expanduser(root)
    base_folder = 'Custom_binary'
    
    # generate train data
    data_train = np.random.normal(mean, var, (data_size_train, in_shape))
    labels_train = np.ones(data_size_train)
    
    file_name_train = 'data_train'
    filepath_train = os.path.join(root, base_folder, file_name_train)
    dataset_train = { "data": data_train, "labels": labels_train}
    pickle.dump(dataset_train, open( filepath_train, "wb" ))  
    
    # generate test data
    data_test = np.random.normal(mean, var, (data_size_test, in_shape))
    labels_test = np.ones(data_size_test)
    
    file_name_test = 'data_test'
    filepath_test = os.path.join(root, base_folder, file_name_test)
    dataset_test = { "data": data_test, "labels": labels_test}
    pickle.dump(dataset_test, open( filepath_test, "wb" )) 



                 
class CUSTOM_BINARY(data.Dataset):
    """ Customized binary classification dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'Custom_binary'
    train_list = ['data_train']
    
    test_list = ['data_test']

    def __init__(self, root, train=True,
                 transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f)
                self.data.append(entry['data'])
                self.targets.append(entry['labels'])

        self.data = np.vstack(self.data).reshape(-1, 1)
        self.targets = np.vstack(self.targets).reshape(-1, 1)

    def __getitem__(self, index): # index = 10
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        val, target = self.data[index], self.targets[index]


        if self.transform is not None:
            val = self.transform(val)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return val, target

    def __len__(self):
        return len(self.data)