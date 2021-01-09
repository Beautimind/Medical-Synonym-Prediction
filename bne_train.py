import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os

class ClassificationNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ClassificationNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.classifier = nn.Sequential(self.fc1, self.fc2, self.fc3)

    def forward(self, input):
        return self.classifier(input)

def test_model(model, x, y, criteria):
    model.eval()
    output = model(x)
    y_np = y.detach().numpy()
    p_np = np.argmax(output.detach().numpy(), axis=1)
    correct_num = np.sum(y_np==p_np)
    accuracy = correct_num / y_np.shape[0]
    loss = criteria(output, valid_tensor_y)
    return accuracy, loss

if __name__ == '__main__':

    conat_dim = 400
    class_num = 2

    # read and preprocess the data
    with open('data/phrases_embedding.txt') as f:
        lines = f.readlines()
    phrase2embed = {}
    embeddings_list = []
    for line in lines:
        elems = line.split('\t')
        vector = np.array([float(num) for num in elems[1].split()])
        phrase2embed[elems[0]] = np.fromstring(vector)
    
    data = pd.read_csv('./data/pairwise_rawdata.csv')
    p1_list = []
    p2_list = []
    l_list = []
    for p1, p2, l in zip(data['name1'], data['name2'], data['label']):
        p1_list.append(phrase2embed[p1])
        p2_list.append(phrase2embed[p2])
        l_list.append(l)

    p1_mtx = np.asarray(p1_list)
    p2_mtx = np.asarray(p2_list)
    l_mtx = np.asarray(l_list)

    
    embedding_dim = 200
    hidden_dim = 1600
    batch_size = 32

    data_mtx = np.concatenate((p1_mtx, p2_mtx), axis=1)

    split = 0.8
    data_num = data_mtx.shape[0]
    train_valid_idx = int(0.8 * data_num)
    train_idx = int(0.8 * train_valid_idx)

    train_x = data_mtx[0:train_idx]
    valid_x = data_mtx[train_idx:train_valid_idx]
    test_x = data_mtx[train_valid_idx:]

    train_y = l_mtx[0:train_idx]
    valid_y= l_mtx[train_idx:train_valid_idx]
    test_y = l_mtx[train_valid_idx:]

    train_dataset = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_tensor_x = torch.from_numpy(valid_x).float()
    valid_tensor_y  = torch.from_numpy(valid_y)
    test_tensor_x = torch.from_numpy(test_x).float()
    test_tensor_y = torch.from_numpy(test_y)

    model = ClassificationNet(data_mtx.shape[1], hidden_dim, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
    criteria = torch.nn.CrossEntropyLoss()
    epoch_num = 10
    best_accuracy = 0.0

    for epoch in range(epoch_num):
        # Train the model What's the behavior of dataloader with suffle=True?
        batch_num = 0
        total_num = 0
        total_correct_num = 0
        total_loss = 0.0
        model.train()
        for x, y in train_dataloader:
            # train the model
            optimizer.zero_grad()
            output = model(x)
            loss = criteria(output, y)
            loss.backward()
            optimizer.step()
            # collect training info
            y_np = y.numpy()
            p_np = np.argmax(output.detach().numpy(), axis=1)

            total_correct_num += np.sum(y_np==p_np)
            total_num += batch_size
            batch_num = batch_num + 1
            total_loss += loss

            if batch_num % 100 == 0:
                msg = 'Batch number {}: '.format(batch_num)
                msg += '[Accumulated Accuracy: {}] [Average Loss: {}]'.format(total_correct_num/total_num, total_loss/total_num)
                print(msg)
            
        #Validate the model
        accuracy, loss = test_model(model, valid_tensor_x, valid_tensor_y, criteria)
        msg = 'Validating result: [Accuracy: {}] [Loss: {}]'.format(accuracy, loss)
        print(msg)
        if accuracy > best_accuracy:
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_accuracy,
                        "loss": loss.item()},
                       os.path.join('./model', "bne.best.tar"))
            print('Model has been saved.')

#Test the model
accuracy, loss = test_model(model, valid_tensor_x, valid_tensor_y, criteria)
msg = 'Testing result: [Accuracy: {}] [Loss: {}]'.format(accuracy, loss)
print(msg)

        
        



    

    


