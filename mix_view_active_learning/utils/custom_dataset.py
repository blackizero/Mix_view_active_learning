import torch
from torch.utils.data.dataset import Dataset
import os
from PIL import Image

class MultiViewDataset(Dataset):
    def __init__(self, root, data_type, transform=None, target_transform=None):
        self.x = []
        self.y = []
        self.root = root

        self.classes, self.class_to_idx = self.find_classes(root)

        self.transform = transform
        self.target_transform = target_transform

        # root/ <label> / <train/test> / <item> / <view>.png
        for label in os.listdir(root):
            for item in os.listdir(root + '/' + label + '/' + data_type): # train or test
                views = []
                for view in os.listdir(root + '/' + label + '/' + data_type + '/' + item):
                    views.append(root + '/' + label + '/' + data_type + '/' + item + '/' + view)

                self.x.append(views)
                self.y.append(self.class_to_idx[label])

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


# override to give pytorch access to any image on the dataset
    def __getitem__(self, index):
        orginal_views = self.x[index]
        views = []

        for view in orginal_views:
            im = Image.open(view)
            im = im.convert('RGB')
            if self.transform is not None:
                im = self.transform(im)
            views.append(im)
        return views, self.y[index]

# override to give pytorch size of dataset
    def __len__(self):
        return len(self.x)
