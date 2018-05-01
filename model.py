import torch
import torch.nn as nn
import torch.nn.functional as F

class SlotRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_classes, bidirectional=False):
        super(SlotRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.batch_size = 1
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.dropout = nn.Dropout(p=0.2)

        self.rnn = nn.GRU(self.hidden_size, hidden_size=self.hidden_size,
                          bidirectional=self.bidirectional,
                          num_layers=2, batch_first=True)

        if bidirectional == True:
            self.linear = nn.Linear(hidden_size*2, n_classes)
        elif bidirectional == False:
            self.linear = nn.Linear(hidden_size, n_classes)

    def forward(self, input):
        input_embedding = self.embedding(input)
        rnn_out, _ = self.rnn(input_embedding, None)
        affine_out = self.linear(torch.squeeze(rnn_out, 0))

        return F.log_softmax(affine_out)

