import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.autograd import Variable

from Models import EncoderBase
import pdb

class LSTMLupiEncoder(EncoderBase):
    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout, embeddings):
        super(LSTMLupiEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings
        self.no_pack_padded_seq = True

        assert num_layers == 2
        self.rnn1 = nn.LSTM(input_size=embeddings.embedding_size,
                                   hidden_size=hidden_size,
                                   num_layers=1,
                                   bidirectional=bidirectional)
        self.rnn1.cuda()

        self.rnn2 = nn.LSTM(input_size=embeddings.embedding_size,
                                   hidden_size=hidden_size,
                                   num_layers=1,
                                   bidirectional=bidirectional)
        self.rnn2.cuda()
 

    def forward(self, input, lengths=None, hidden=None, dropout_mask=None):
        """ See EncoderBase.forward() for description of args and returns."""
        self._check_args(input, lengths, hidden)
        emb = self.embeddings(input)
        s_len, batch, emb_dim = emb.size()
        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            # This is not supported yet
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)

        if hidden is not None:
            outputs0, hidden_t0 = self.rnn1(packed_emb, hidden[0])
            mask = Variable(torch.Tensor(outputs0.size(0), outputs0.size(1), 500).fill_(0.3).bernoulli_()).cuda()
            if self.training:
                outputs0 = outputs0.mul(mask)
            outputs1, hidden_t1 = self.rnn2(outputs0,hidden[1])
        else:
            outputs0, hidden_t0 = self.rnn1(packed_emb, None)
            mask = Variable(torch.Tensor(outputs0.size(0), outputs0.size(1), 500).fill_(0.3).bernoulli_()).cuda()
            if self.training:
                outputs0 = outputs0.mul(mask)
            outputs1, hidden_t1 = self.rnn2(outputs0,None)

        if lengths is not None and not self.no_pack_padded_seq:
            outputs = unpack(outputs)[0]
        two_layer_hidden = (torch.cat((hidden_t0[0],hidden_t1[0]),0),torch.cat((hidden_t0[1], hidden_t1[1]),0))
        return two_layer_hidden, outputs1

