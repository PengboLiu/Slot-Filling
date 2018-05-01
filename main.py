import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import progressbar
from lstm_crf import BiLSTM_CRF
from model import SlotRNN
from evaluate import conlleval
import copy


class Config:
    cuda = False 
    rnn_bidirectional = True
    hidden_size = 256
    n_epochs = 20
    learning_rate = 0.01
    embedding_size = 256

# Preprocess Toolbox
def extract_nested_column(file_path, col_idx):
    s = open(file_path, 'r').read().split('\n\n')

    nested_col = []

    for i in range(len(s)):
        nested_col.append([pair.split('\t')[col_idx] for pair in s[i].split('\n')])
    return nested_col

def unnest_list(nested_list):
    unnested_list = []
    for i in range(len(nested_list)):
        for j in range(len(nested_list[i])):
            unnested_list.append(nested_list[i][j])

    return unnested_list

def allocate_index(item_list):
    idx_dict = {}
    idx = 0
    for each_item in item_list:
        if each_item not in idx_dict:
            idx_dict[each_item] = idx
            idx += 1
    return idx_dict

def map_index(item_nested_list, idx_dict):
    idx_nested_list = copy.deepcopy(item_nested_list)   
    for i in range(len(item_nested_list)):
        for j in range(len(item_nested_list[i])):
            idx_nested_list[i][j] = idx_dict[item_nested_list[i][j]]
    return idx_nested_list

def preprocess(file_path, idx_dict=None):
    words = extract_nested_column(file_path, 0)
    labels = extract_nested_column(file_path, 1)

    if idx_dict == None:
        words_idx_dict = allocate_index(unnest_list(words))
        labels_idx_dict = allocate_index(unnest_list(labels))
    else:
        words_idx_dict = idx_dict[0]
        labels_idx_dict = idx_dict[1]

    # 数值化
    words_idx = map_index(words, words_idx_dict)
    labels_idx = map_index(labels, labels_idx_dict)

    return list2ndarray(words), list2ndarray(labels), list2ndarray(words_idx), \
           list2ndarray(labels_idx), words_idx_dict, labels_idx_dict

def list2ndarray(nested_list):
    for i in range(len(nested_list)):
        nested_list[i] = np.array(nested_list[i])
    return nested_list


def var2np(variable):
    return torch.max(variable, 1)[1].data.cpu().squeeze(-1).numpy()

def main():
    # Load data
    # Load words_idx_dict, labels_idx_dict from atis.all.txt.
    words_idx_dict, labels_idx_dict = preprocess('./data/atis.all.txt')[4:6]
    # Load train data, including words, label, words_id, label_id.
    train_words, train_labels, train_x, train_label = \
        preprocess('./data/atis.train.txt', (words_idx_dict, labels_idx_dict))[0:4]
    # Load valid data, including words, label, words_id, label_id.
    val_words, val_labels, val_x, val_label = \
        preprocess('./data/atis.test.txt', (words_idx_dict, labels_idx_dict))[0:4]

    # id2word & id2label
    idx2w = {words_idx_dict[k]: k for k in words_idx_dict}
    idx2la = {labels_idx_dict[k]: k for k in labels_idx_dict}
    # vocab number & label number
    n_vocab = len(words_idx_dict)
    n_label = len(labels_idx_dict)
    print("vocab_size:{}, label_size:{}".format(n_vocab, n_label))

    # define model
    model = SlotRNN(n_vocab, Config.hidden_size, n_label, bidirectional=Config.rnn_bidirectional)

    # model = BiLSTM_CRF(Config.embedding_size, n_vocab, Config.hidden_size, n_label)

    if Config.cuda == True:
        model.cuda()

    print(model)
    # id->words & id->labels
    words_train = [list(map(lambda x: idx2w[x], w)) for w in train_x]
    groundtruth_train = [list(map(lambda x: idx2la[x], y)) for y in train_label]

    words_val = [list(map(lambda x: idx2w[x], w)) for w in val_x]
    groundtruth_val = [list(map(lambda x: idx2la[x], y)) for y in val_label]

    optimizer = optim.SGD(model.parameters(), lr=Config.learning_rate)
    criterion = nn.NLLLoss()

    # train
    for i in range(Config.n_epochs):
        print("Epoch {}".format(i))
        print("Training =>")

        train_pred_label = []
        avgLoss = 0
        bar = progressbar.ProgressBar(maxval=len(train_x))
        for n_batch, sent in bar(enumerate(train_x)):
            optimizer.zero_grad()
            label = train_label[n_batch]
            label_tensor = Variable(torch.LongTensor(label))
            
            if(Config.cuda == True):
                label_tensor = label_tensor.cuda()
            sent = sent[np.newaxis, :]
            sent = Variable(torch.LongTensor(sent))
            if(Config.cuda == True):
                sent = sent.cuda()
            
            pred = model(sent)
            train_pred_label.append(var2np(pred))
            loss = criterion(pred, label_tensor)
            avgLoss += loss.data[0]
            loss.backward()
            optimizer.step()

        avgLoss = avgLoss/n_batch
        train_pred = [list(map(lambda x: idx2la[x], y)) for y in train_pred_label]
        print("btach = {},loss = {}".format(i, avgLoss))
        eval(model, groundtruth_train, words_train, pred_res=train_pred)
        eval(model, groundtruth_val, words_val, val=val_x, idx2la=idx2la)


def eval(model, groundtruth, words, val=None, idx2la=None, pred_res=None):
    model.eval()
    if pred_res is None:
        print("Test =>")
        pred_label = []
        bar = progressbar.ProgressBar(maxval=len(val))
        for n_batch, sent in bar(enumerate(val)):
            sent = sent[np.newaxis, :]
            sent = Variable(torch.LongTensor(sent))
            if(Config.cuda == True):
                sent = sent.cuda()
            pred = model(sent)
            pred_label.append(var2np(pred))
        pred_res = [list(map(lambda x: idx2la[x], y)) for y in pred_label]
    model.train()
    
    print(conlleval(pred_res, groundtruth, words, 'tmp.txt'))

if __name__ == '__main__':

    main()