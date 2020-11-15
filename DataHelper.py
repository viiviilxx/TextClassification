import numpy as np
import torch, os, re
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
import pickle


cpu = torch.device('cpu')


def load_Data(path) :
    print('loading ' + path + '...', end = '', flush = True)
    with open(path, 'rb') as file :
        document = pickle.load(file)
    print('done!', flush = True)
    return list(document[0]), list(document[1])


class BertHelper(Dataset) :
    def __init__(self, path) :
        super(BertHelper, self).__init__()
        self.text_ids, self.label_vectors = load_Data(path)
        self.text_ids = torch.tensor(self.text_ids, device = cpu)
        self.label_vectors = torch.tensor(self.label_vectors, device = cpu, dtype = torch.float32)


    def __len__(self) :
        return self.text_ids.size()[0]


    def __getitem__(self, index) :
        text = self.text_ids[index]
        label = self.label_vectors[index]
        return text, label


    def getLabelNumpy(self) :
        return self.label_vectors.numpy().copy()

