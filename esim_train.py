from __future__ import division
import sys
from os.path import expanduser
import torch
import numpy
from esim_model import ESIM, generate_mask
import time
from datetime import timedelta
from torch.utils.data import DataLoader
from esim_data import SynonymDataset
import pandas as pd
import pickle
from torch.autograd import Variable
import os
from tqdm import tqdm

# batch preparation
def prepare_data(seqs_x, seqs_y, labels, maxlen=None):
	lengths_x = [len(s) for s in seqs_x]
	lengths_y = [len(s) for s in seqs_y]

	if maxlen is not None:
		new_seqs_x = []
		new_seqs_y = []
		new_lengths_x = []
		new_lengths_y = []
		new_labels = []
		for l_x, s_x, l_y, s_y, l in zip(lengths_x, seqs_x, lengths_y, seqs_y, labels):
			if l_x < maxlen and l_y < maxlen:
				new_seqs_x.append(s_x)
				new_lengths_x.append(l_x)
				new_seqs_y.append(s_y)
				new_lengths_y.append(l_y)
				new_labels.append(l)
		lengths_x = new_lengths_x
		seqs_x = new_seqs_x
		lengths_y = new_lengths_y
		seqs_y = new_seqs_y
		labels = new_labels

		if len(lengths_x) < 1 or len(lengths_y) < 1:
			return None, None, None, None, None

	n_samples = len(seqs_x)
	maxlen_x = numpy.max(lengths_x)
	maxlen_y = numpy.max(lengths_y)

	x = numpy.zeros((maxlen_x, n_samples)).astype('int64')
	y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
	x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
	y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
	l = numpy.zeros((n_samples,)).astype('int64')
	for idx, [s_x, s_y, ll] in enumerate(zip(seqs_x, seqs_y, labels)):
		x[:lengths_x[idx], idx] = s_x
		x_mask[:lengths_x[idx], idx] = 1.
		y[:lengths_y[idx], idx] = s_y
		y_mask[:lengths_y[idx], idx] = 1.
		l[idx] = ll

	if torch.cuda.is_available():
		x=Variable(torch.LongTensor(x)).cuda()
		x_mask=Variable(torch.Tensor(x_mask)).cuda()
		y=Variable(torch.LongTensor(y)).cuda()
		y_mask=Variable(torch.Tensor(y_mask)).cuda()
		l=Variable(torch.LongTensor(l)).cuda()
	else:
		x = Variable(torch.LongTensor(x))
		x_mask = Variable(torch.FloatTensor(x_mask))
		y = Variable(torch.LongTensor(y))
		y_mask = Variable(torch.FloatTensor(y_mask))
		l = Variable(torch.LongTensor(l))
	return x, x_mask, y, y_mask, l

# some utilities
def ortho_weight(ndim):
	"""
	Random orthogonal weights

	Used by norm_weights(below), in which case, we
	are ensuring that the rows are orthogonal
	(i.e W = U \Sigma V, U has the same
	# of rows, V has the same # of cols)
	"""
	W = numpy.random.randn(ndim, ndim)
	u, s, v = numpy.linalg.svd(W)
	return u.astype('float32')

def setup_vocab(data):
  all_words = set()
  for n1, n2 in zip(data['name1'], data['name2']):
    all_words.update(n1.split())
    all_words.update(n2.split())
  all_words = list(all_words)
  all_words.insert(0, '<pad>')
  word2id = {}
  id = 0
  for id, word in enumerate(all_words):
    word2id[word] = id
  
  return all_words, word2id

def format_data(data, word2id):
  data_list = []
  for n1, n2, label in zip(data['name1'], data['name2'], data['label']):
    row = []
    ws1 = n1.split()
    ws2 = n2.split()
    id1_list = [word2id[w] for w in ws1]
    id2_list = [word2id[w] for w in ws2]
    row.append(id1_list)
    row.append(id2_list)
    row.append(label)
    data_list.append(row)
  return data_list

def collate(data):
	x1_list, x2_list, label_list = [], [], []
	for row in data:
		x1_list.append(row[0])
		x2_list.append(row[1])
		label_list.append(row[2])
	return x1_list, x2_list, label_list

data = pd.read_csv('./data/pairwise_data.csv')

#build the word-id mapping
all_words, word2id = setup_vocab(data)

#build the embedding matrix
with open('./data/embeddings.pickle', 'rb') as f_embed:
  word2vec = pickle.load(f_embed)
pretrained_emb = []
for word in all_words:
  if word in word2vec:
    pretrained_emb.append(word2vec[word])
  else:
    pretrained_emb.append(numpy.random.uniform(-0.05, 0.05, size=[200])) 
embeddings = numpy.stack(pretrained_emb)

formatted_data = format_data(data, word2id)
size = len(formatted_data)
split1 = int(size * 0.8)
split2 = int(split1 * 0.8)
train_pairs = formatted_data[0 : split2]
dev_pairs = formatted_data[split2 : split1]
test_pairs = formatted_data[split1 :]

train_dataset = SynonymDataset(train_pairs)
train_dataloader = DataLoader(train_dataset, batch_size=32, collate_fn=collate)
valid_dataset = SynonymDataset(dev_pairs)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, collate_fn=collate)
test_dataset = SynonymDataset(test_pairs)
test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=collate)

n_words = len(all_words)
dim_word = 200

criterion = torch.nn.CrossEntropyLoss()
model = ESIM(dim_word, 2, n_words, dim_word, embeddings)
if torch.cuda.is_available():
	model = model.cuda()
	criterion = criterion.cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0004)
print('start training...')
accumulated_loss=0
batch_counter=0
report_interval = 100
best_dev_loss=10e10
best_dev_loss2=10e10
clip_c=10
max_len=100
max_result=0
best_accuracy = 0.0
num_epochs = 10
model.train()
for epoch in range(num_epochs):
	accumulated_loss = 0
	model.train()
	start_time = time.time()
	train_sents_scaned = 0
	train_num_correct = 0
	batch_counter=0
	for x1, x2, y in train_dataloader:
		x1, x1_mask, x2, x2_mask, y = prepare_data(x1, x2, y, maxlen=max_len)
		train_sents_scaned += len(y)
		optimizer.zero_grad()
		output = model(x1, x1_mask, x2, x2_mask)
		result = output.data.cpu().numpy()
		a = numpy.argmax(result, axis=1)
		b = y.data.cpu().numpy()
		train_num_correct += numpy.sum(a == b)
		loss = criterion(output, y)
		loss.backward()
		''''''
		grad_norm = 0.

		for m in list(model.parameters()):
			grad_norm+=m.grad.data.norm() ** 2

		for m in list(model.parameters()):
			if grad_norm>clip_c**2:
				try:
					m.grad.data= m.grad.data / torch.sqrt(grad_norm) * clip_c
				except:
					pass
		''''''
		optimizer.step()
		accumulated_loss += loss.item()
		batch_counter += 1
		if batch_counter % report_interval == 0:
			print('--' * 20)
			msg = '%d completed epochs, %d batches' % (epoch, batch_counter)
			msg += '\t Average Loss: %f' % (accumulated_loss / train_sents_scaned)
			msg += '\t Accuracy: %f' % (train_num_correct / train_sents_scaned)
			print(msg)
	
	# Start validate
	print('Start to validate model...')
	model.eval()
	valid_loss = 0.0
	valid_correct_num = 0
	for x1, x2, y in tqdm(valid_dataloader):
		x1, x1_mask, x2, x2_mask, y = prepare_data(x1, x2, y, maxlen=max_len)
		output = model(x1, x1_mask, x2, x2_mask)
		result = output.numpy()
		a = numpy.argmax(result, axis=1)
		b = y.numpy()
		valid_correct_num += numpy.sum(a == b)
		loss = criterion(output, y)
		valid_loss += loss.item()

	valid_accuracy = valid_correct_num / len(valid_dataloader.dataset)
	elapsed_time = time.time() - start_time
	if valid_accuracy > best_accuracy:
		best_accuracy = valid_accuracy
		print('--' * 20)
		msg = 'Validating result:\n'
		msg += 'Time for this Epoch: %ds' % (elapsed_time)
		msg += '\t Accuracy: %f' % (valid_accuracy)
		msg += '\t Average Loss: %f' % (valid_loss / len(valid_dataloader.dataset))
		print(msg)
		torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_accuracy,},
                       os.path.join('./model', "best.esim.tar"))

# Testing the model
print('training is done, start testing the model')
model.eval()
test_loss = 0.0
test_correct_num = 0
start_time = time.time()
for x1, x2, y in tqdm(test_dataloader):
	x1, x1_mask, x2, x2_mask, y = prepare_data(x1, x2, y, maxlen=max_len)
	output = model(x1, x1_mask, x2, x2_mask)
	result = output.data.cpu().numpy()
	a = numpy.argmax(result, axis=1)
	b = y.data.cpu().numpy()
	test_correct_num += numpy.sum(a == b)
	loss = criterion(output, y)
	test_loss += loss.item()
test_accuracy = test_correct_num / len(test_dataloader.dataset)
elapsed_time = time.time() - start_time
print('--' * 20)
msg = 'Testing result:\n'
msg += 'Time Using: %ds' % (elapsed_time)
msg += '\t Accuracy: %f' % (test_accuracy)
msg += '\t Average Loss: %f' % (test_loss / len(test_dataloader.dataset))
print(msg)

