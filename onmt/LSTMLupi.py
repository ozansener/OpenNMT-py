import torch

from Models import EncoderBase

class LSTMLupiEncoder(EncoderBase)
    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout, embeddings):
        super(LSTMLupiEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings
        self.no_pack_padded_seq = False

        # We will put the dropout between two layers
        self.rnns = {}
        for layer in range(num_layers):
            self.rnn[layer] = LSTM(input_size=embeddings.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    bidirectional=bidirectional)

    def forward(self, input, lengths=None, hidden=None, dropout_mask=None):
        """ See EncoderBase.forward() for description of args and returns."""
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)

        if hidden is not None:
            outputs0, hidden_t0 = self.rnn[0](packed_emb, hidden[0])
            outputs0 = outputs0.mul(dropout_mask)
            outputs1, hidden_t1 = self.rnn[1](outputs0,hidden[1])
        else:
            outputs0, hidden_t0 = self.rnn[0](packed_emb, None)
            outputs0 = outputs0.mul(dropout_mask)
            outputs1, hidden_t1 = self.rnn[1](outputs0,None)

        if lengths is not None and not self.no_pack_padded_seq:
            outputs = unpack(outputs)[0]

        return torch.stack([hidden_t0,hidden_t1]), torch.stack([outputs0, outputs1])

