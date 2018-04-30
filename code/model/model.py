from base import BaseModel
import torch.nn as nn

class RNNLM(BaseModel):
    def __init__(self, config):
        super(RNNLM, self).__init__(config)
        self.config = config
        self.rnn_type = self.config['rnn_type']
        self.ninput = self.config['ninput']
        self.nhidden = self.config['nhidden']
        self.nlayers = self.config['nlayers']
        self.ntoken = self.config['ntoken']
        self.dropout = self.config['dropout']
        self.tie_weights = self.config['tie_weights']
        self.drop = nn.Dropout(self.dropout)
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        if self.rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, self.rnn_type)(self.ninput, self.nhidden, self.nlayers, dropout=self.dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(self.ninput, self.nhidden, self.nlayers, nonlinearity=nonlinearity, dropout=self.dropout)
        self.decoder = nn.Linear(self.nhidden, self.ntoken)

        if self.tie_weights:
            if self.nhidden != self.ninput:
                raise ValueError('When using the tied flag, nhidden must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = self.rnn_type
        self.nhidden = self.nhidden
        self.nlayers = self.nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        input = input.t().contiguous()
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