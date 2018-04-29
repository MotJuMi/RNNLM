from base import BaseModel
import torch.nn as nn

class RNNLM(BaseModel):
    def __init__(self, config, rnn_type, ntoken,
                 ninput, nhidden, nlayers, dropout=0.5,
                 tie_weights=False):
        super(RNNLM, self).__init__(config)
        self.config = config
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninput)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninput, nhidden, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninput, nhidden, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhidden, ntoken)

        if tie_weights:
            if nhidden != ninput:
                raise ValueError('When using the tied flag, nhidden must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhidden = nhidden
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data_uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, batch_size, self.nhidden),
                    weight.new_zeros(self.nlayers, batch_size, self.nhidden))
        else:
            return weight.new_zeros(self.nlayers, batch_size, self.nhidden)