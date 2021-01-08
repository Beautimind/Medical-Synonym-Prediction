import torch.nn as nn
import numpy as np
import pandas as pd

class ClassificationNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.classifier = nn.Sequential(self.fc1, self.fc2, self.fc3)

    def forward(self, input):
        return self.classifier(input)

if __name__ == '__main__':

    conat_dim = 400
    class_num = 2

    # read and preprocess the data
    with open('data/phrases_embedding.txt') as f:
        lines = f.readlines()
    phrase2embed = {}
    for line in lines:
        elems = lines.split('\t')
        phrase2embed[elems[0]] = np.fromstring(elems[1])
    
    data = pd.read_csv('./data/pairwise_data.csv')
    p1_list = []
    p2_list = []
    l_list = []
    for p1, p2, l in zip(data['name1'], data['name2'], data['label']):
        p1_list.append(p1)
        p2_list.append(p2)
        l_list.append(l)

    p1_mtx = np.asarray(p1_list)
    p2_mtx = np.asarray(p2_list)
    l_mtx = np.asarray(l_list)

    


