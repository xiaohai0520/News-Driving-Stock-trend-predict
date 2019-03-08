import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from dataset import embedding_dim


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, T):
        """
        input size: number of underlying factors (25 )
        T: number of time steps (10)
        hidden_size: dimension of the hidden state
        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T

        self.news_linear = nn.Linear(in_features=embedding_dim, out_features=1)
        self.attn1 = nn.Linear(in_features=2 * hidden_size, out_features=self.T)
        self.attn2 = nn.Linear(in_features=self.T, out_features=self.T)
        self.tanh = nn.Tanh()
        self.attn3 = nn.Linear(in_features=self.T, out_features=1)
        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)

    def forward(self, input_data):
        #batch size :128
        batch_size = input_data.size(0)
        # print(input_data.shape)

        #linear change: 128 * 10 * 25 * 768 -> 128 * 10 * 25 * 1
        input_data = self.news_linear(input_data)
        # print(input_data.shape)

        #squeeze  128 * 10 * 25 * 1 -> 128 * 10 * 25
        # input_data: (batch_size, T , input_size)
        input_data = input_data.squeeze(3)
        # print(input_data.shape)

        # input_weighted: 128 * 10 * 25   input_encoded : 128 * 25 * 64
        # input_weighted(batch_size, T, input_size)     input_encoded(batch_size, T, hidden_size)
        input_weighted = Variable(torch.zeros(batch_size, self.T, self.input_size))
        input_encoded = self.init_variable(batch_size,self.T,self.hidden_size)

        # hidden, cell: initial states with dimension hidden_size
        # hidden: 1 * 128 * 64   cell: 1 * 128 * 64
        # hidden = cell : 1 * batch_size * hidden_size
        hidden = self.init_variable(1, batch_size,self.hidden_size)
        cell = self.init_variable(1, batch_size, self.hidden_size)


        for t in range(self.T):

            # Eqn. 8: concatenate the hidden states and the cell states

            # 128 * 10 * 64 + 128 * 10 * 64 = 128 * 10 * 128
            # batch_size * T * (2 * hidden_size)
            x = torch.cat((self.embedding_hidden(hidden), self.embedding_hidden(cell)), dim=2)

            # batch_size * input_size * (2 * hidden_size) -> batch_size * input_size * T
            # 128 * 25 * 128 -> (z1) 128 * 25 * 10
            z1 = self.attn1(x)
            # print('z1:',z1.shape)

            # batch_size * input_size * T  ->  batch_size * input_size * T
            #  128 * 25 * 10  -> 128 * 25 * 10
            z2 = self.attn2(input_data.permute(0, 2, 1))
            # print('z2:', z2.shape)

            # batch_size * input_size * T
            #x: 128 * 25 * 10
            x = z1 + z2

            # batch_size * input_size * T  -> batch_size * input_size * 1
            # 128 * 25 * 10 -> 128 * 25 * 1
            z3 = self.attn3(self.tanh(x))

            # Eqn. 9: Softmax the attention weights
            # (batch_size, input_size)  128 * 25
            if batch_size > 1:
                attn_weights = F.softmax(z3.view(batch_size, self.input_size), dim=1)
            else:
                attn_weights = self.init_variable(batch_size, self.input_size) + 1


            # attn_weights = F.softmax(x.view(-1, self.input_size), dim=1)

            # Eqn. 10: update driving series
            # (batch_size, input_size)   128 *25
            weighted_input = torch.mul(attn_weights, input_data[:, t, :])

            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            # self.lstm_layer.flatten_parameters()

            # Eqn. 11: LSTM
            # weighted_input.unsqueeze(0) : 1 * 128 * 25
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]
            # Save output

            # input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden

        return input_encoded

    def embedding_hidden(self, x):
        return x.repeat(self.input_size, 1, 1).permute(1, 0, 2)

    def init_variable(self, *args):
        zero_tensor = torch.zeros(args)
        if torch.cuda.is_available():
            zero_tensor = zero_tensor.cuda()
        return Variable(zero_tensor)


class Decoder(nn.Module):

    def __init__(self, encoder_hidden_size, decoder_hidden_size, T, out_feats=2):
        super(Decoder, self).__init__()


        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.T = T

        self.attn1 = nn.Linear(in_features=2*decoder_hidden_size, out_features=encoder_hidden_size)
        self.attn2 = nn.Linear(in_features=encoder_hidden_size,out_features=encoder_hidden_size)
        self.tanh = nn.Tanh()
        self.attn3 = nn.Linear(in_features=encoder_hidden_size, out_features=1)

        self.lstm_layer = nn.LSTM(input_size=1, hidden_size=decoder_hidden_size)
        self.tilde = nn.Linear(in_features=encoder_hidden_size + 1, out_features=1)
        self.fc1 = nn.Linear(in_features=decoder_hidden_size + encoder_hidden_size,out_features=decoder_hidden_size)
        self.fc2 = nn.Linear(in_features=decoder_hidden_size,out_features=out_feats)
        # self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_seq):
        #batch_size : 128
        batch_size = input_encoded.size(0)

        # hidden : 1 * batch_size * decode hidden size    1 * 128 * 64
        hidden = self.init_variable(1,batch_size,self.decoder_hidden_size)

        # cell : 1 * batch_size * decode hidden size    1 * 128 * 64
        cell = self.init_variable(1, batch_size, self.decoder_hidden_size)

        # context : batch_size * decode_hidden   128 * 64
        context = self.init_variable(batch_size, self.decoder_hidden_size)

        for t in range(self.T):

            # Eqn. 12: concatenate the hidden states and the cell states
            # 128 * 10 * 64 + 128 * 10 * 64 = 128 * 10 * 128
            # batch_size * T * (2 * hidden_size)
            x = torch.cat((self.embedding_hidden(hidden),self.embedding_hidden(cell)),dim=2)

            # batch_size * T * (2 * decode_hidden_size) -> batch_size * T * encode_hidden_size
            # 128 * 10 * 128 -> (z1) 128 * 10 * 64
            z1 = self.attn1(x)

            # input_encoded: 128 * 10 * 64   batch_size * T * hidden_size
            # output not change   128 * 10 * 64
            z2 = self.attn2(input_encoded)

            # z1 + z2   128 * 10 * 64 + 128 * 10 * 64 = 128 * 10 * 64
            x = z1 + z2

            # encode_hidden_size -> 1
            # 128 * 10 * 64 -> 128 * 10 * 1
            z3 = self.attn3(self.tanh(x))

            # Eqn. 13: softmax to get weight
            # x.view(): 128 * 10   weight 128 * 10
            # beta_weight = F.softmax(x.view(batch_size, -1),dim=1)

            if batch_size > 1:
                beta_weight = F.softmax(z3.view(batch_size, -1), dim=1)
            else:
                beta_weight = self.init_variable(batch_size, self.decode_hidden_size) + 1

            # Eqn. 14: sum to new context with tempial weight
            # 128 * 1 * 10   bmm   128 * 10 * 64   = 128 * 1 * 64
            # 128 * 1 * 64 -> 128 * 64
            context = torch.bmm(beta_weight.unsqueeze(1),input_encoded).squeeze(1)

            # Eqn. 15
            if t < self.T -1:

                # yc: 128 * (1 + 64) -> 128 * 65
                yc = torch.cat((y_seq[:, t].unsqueeze(1), context), dim=1)

                # (batch_size, out_size)   128 * 1
                y_tilde = self.tilde(yc)

                # Eqn. 16: LSTM
                # self.lstm_layer.flatten_parameters()

                # y_tilde.unsqueeze:  1 * 128 * 1  hidden and cell : 1 * 128 * 64
                _, lstm_state = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))

                # 1 * batch_size * decoder_hidden_size
                hidden = lstm_state[0]

                # 1 * batch_size * decoder_hidden_size
                cell = lstm_state[1]

        # Eqn. 22: final output
        # hidden.squeeze : 128 *64(batch_size * decoder_hidden_size)
        # context : 128 * 64   batch_size * decoder_hidden_size
        # fc1 cat 128 * 128 -> 128 * 64
        # fc2 128 * 64 -> 128 * 2

        return self.fc2(self.fc1(torch.cat((hidden.squeeze(0), context), dim=1)))
        # return F.softmax(x,dim=1)


    def init_variable(self, *args):
        zero_tensor = torch.zeros(args)
        if torch.cuda.is_available():
            zero_tensor = zero_tensor.cuda()
        return Variable(zero_tensor)

    def embedding_hidden(self, x):
        return x.repeat(self.T, 1, 1).permute(1, 0, 2)