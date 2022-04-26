import os
import os.path as osp
import numpy as np
from torch.utils import data

__all__ = ['EPIDataSet']


class EPIDataSet(data.Dataset):
    def __init__(self, data, label):
        super(EPIDataSet, self).__init__()
        self.data = data
        self.label = label

        assert len(self.data) == len(self.label), \
            "the number of sequences and labels must be consistent."

        print("The number of positive data is {}".format(sum(self.label.reshape(-1) == 1)))
        print("The number of negative data is {}".format(sum(self.label.reshape(-1) == 0)))
        print("pre-process data is done.")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_one = self.data[index]
        label_one = self.label[index]

        return {"data": data_one, "label": label_one}

