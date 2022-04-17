import os
import sys
import numpy as np
import random
import pickle

# image preprocess
import cv2

# torch
import torch


class Feeder(torch.utils.data.Dataset):
    """ Feeder for H A C K
    Arguments:
        label_path: the path to label ('.pkl' data)
        image_path: the path to file which contains image path ('.pkl' data)
        debug: If true, only use the first 100 samples
        ...
    """
    def __init__(
        self,
        label_path,
        prototype_path,
        sample_weights,
        image_path=None,
        imagepath=False,
        image=True,
        image_size=256,
        istrain=False,
        debug=False,
        mmap=False,
        num_frame=8,
        **kwargs,
    ):

        self.debug = debug
        self.label_path = label_path
        self.image_path = image_path
        self.image = image
        self.imagepath = imagepath
        self.image_size = image_size
        self.istrain = istrain
        self.num_frame = num_frame
        self.loader_scale = num_frame

        self.sample_weights = np.array(sample_weights)
        self.prototypes = np.load(prototype_path)
        self.num_prototype = len(self.prototypes)

        self.load_data(mmap)

    def load_data(self, mmap):

        with open(self.label_path, 'rb') as f:
            _, self.label = pickle.load(f)
        self.label = np.array(self.label).squeeze()

        if self.debug:
            sample_idxs = random.sample(list(range(len(self.label))), 2000)
            self.label = self.label[sample_idxs]
            self.data = self.data[sample_idxs]

        if self.image or self.imagepath:
            # load image
            with open(self.image_path, 'rb') as f:
                _, self.imagepaths = pickle.load(f)

            self.imagepaths = np.array(self.imagepaths).squeeze()

        self.total_frame = len(self.label)
        print("  data.shape", self.data.shape)
        print("  imagepaths", len(self.imagepaths), len(self.imagepaths[0]))
        print("  label", len(self.label), len(self.label[0]))

    def __len__(self):
        return int(self.total_frame / self.loader_scale)

    def __getitem__(self, index):

        if self.istrain:
            frameidxs = [
                random.randint(0, self.total_frame - 1) for i in range(self.num_frame)
            ]
        else:
            frameidxs = [index]

        # get data
        label = []
        image = []
        prototype_id = []
        prototype_label = []
        for fi, frameidx in enumerate(frameidxs):
            target = self.label[frameidx][np.newaxis, :]
            dists = np.linalg.norm(
                np.repeat(target, self.num_prototype, axis=0) -
                self.prototypes,
                ord=1,
                axis=1)
            min_idxs = (dists == dists.min()).nonzero()[0]
            prob = self.sample_weights[min_idxs]
            prob = prob / sum(prob)
            proto_label = np.zeros(self.num_prototype)
            proto_label[min_idxs] = 1

            label.append(self.label[frameidx])
            prototype_label.append(proto_label)
            if dists.min() == 0:
                prototype_id.append(dists.argmin().squeeze())
            else:
                prototype_id.append(-1)

            if self.image:
                img = cv2.imread(self.imagepaths[frameidx].replace(
                    "/home/ubuntu/Documents", "../..",
                    1).replace("../../..", "../..", 1))
                face = cv2.resize(img, (self.image_size, self.image_size))
                face = face.transpose((2, 0, 1))
                image.append(face / 255.0)

        data_numpy = np.array(data_numpy)
        label = np.array(label)
        image = np.array(image)
        prototype_label = np.array(prototype_label)
        prototype_id = np.array(prototype_id)

        if self.imagepath:
            imgpath = self.imagepaths[frameidx]

        if self.image and not self.imagepath:
            return data_numpy, label, image, prototype_label, prototype_id
        elif self.imagepath:
            return data_numpy, label, image, prototype_label, prototype_id, imgpath
        else:
            return data_numpy, label, prototype_label, prototype_id
