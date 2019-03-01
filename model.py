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
        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.attn_linear = nn.Linear(in_features=2 * hidden_size + T, out_features=1)

    def forward(self, input_data):
        # print(input_data.shape)
        input_data = self.news_linear(input_data)
        # print(input_data.shape)
        input_data = input_data.squeeze(3)
        # print(input_data.shape)
        # input_data: (batch_size, T , input_size)
        input_weighted = Variable(torch.zeros(input_data.size(0), self.T, self.input_size))
        input_encoded = Variable(torch.zeros(input_data.size(0), self.T, self.hidden_size))
        # hidden, cell: initial states with dimension hidden_size
        hidden = self.init_hidden(input_data, self.hidden_size)  # 1 * batch_size * hidden_size
        cell = self.init_hidden(input_data, self.hidden_size)

        for t in range(self.T):
            # Eqn. 8: concatenate the hidden states with each predictor
            hidden_change = hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2)
            # print(hidden.size())
            cell_change = cell.repeat(self.input_size, 1, 1).permute(1, 0, 2)
            # print(cell.size())
            input_data_change = input_data.permute(0, 2, 1)
            # print(input_data.size())
            x= torch.cat((hidden_change,cell_change,input_data_change),dim=2)
            # x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
            #                cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
            #                input_data.permute(0, 2, 1)), dim=2)  # batch_size * input_size * (2*hidden_size + T )
            # Eqn. 8: Get attention weights
            # print(x.size())
            x = self.attn_linear(x.view(-1, self.hidden_size * 2 + self.T))  # (batch_size * input_size) * 1
            # print(x.size())
            # Eqn. 9: Softmax the attention weights
            attn_weights = F.softmax(x.view(-1, self.input_size), dim=1)  # (batch_size, input_size)
            # print(attn_weights.size())
            # Eqn. 10: LSTM
            # print('1',input_data[:,t,:].size())
            weighted_input = torch.mul(attn_weights, input_data[:, t, :])  # (batch_size, input_size)
            # print(weighted_input.size())
            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]
            # Save output

            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden

        return input_weighted, input_encoded

    def init_hidden(self,x, hidden_size):
        if torch.cuda.is_available():
            return Variable(torch.zeros(1, x.size(0), hidden_size)).cuda()
        else:
            return Variable(torch.zeros(1, x.size(0), hidden_size))



class Decoder(nn.Module):

    def __init__(self, encoder_hidden_size, decoder_hidden_size, T, out_feats=2):
        super(Decoder, self).__init__()

        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size,
                                                  encoder_hidden_size),
                                        nn.Tanh(),
                                        nn.Linear(encoder_hidden_size, 1))
        self.lstm_layer = nn.LSTM(input_size=out_feats, hidden_size=decoder_hidden_size)
        self.fc = nn.Linear(encoder_hidden_size + T, out_feats)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, out_feats)

        self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_history):
        # input_encoded: (batch_size, T - 1, encoder_hidden_size)
        # y_history: (batch_size, (T-1))
        # Initialize hidden and cell, (1, batch_size, decoder_hidden_size)

        # ？？？？？？why not on gpu
        if torch.cuda.is_available():
            input_encoded = input_encoded.cuda()
        hidden = self.init_hidden(input_encoded, self.decoder_hidden_size)
        cell = self.init_hidden(input_encoded, self.decoder_hidden_size)
        context = Variable(torch.zeros(input_encoded.size(0), self.encoder_hidden_size))

        for t in range(self.T):
            # (batch_size, T, (2 * decoder_hidden_size + encoder_hidden_size))

            hidden_change = hidden.repeat(self.T, 1, 1).permute(1, 0, 2)
            # print(hidden.size())
            cell_change = cell.repeat(self.T, 1, 1).permute(1, 0, 2)
            # print(cell.size())
            # input_data_change = input_encoded.permute(0, 2, 1)

            x = torch.cat((hidden_change,cell_change,input_encoded), dim=2)
            # Eqn. 12 & 13: softmax on the computed attention weights
            x = F.softmax(
                self.attn_layer(
                    x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)
                ).view(-1, self.T),
                dim=1)  # (batch_size, T )
            # print("x1",x)
            # Eqn. 14: compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded).squeeze(1)  # (batch_size, encoder_hidden_size)
            print('content',context.shape)
            # print(y_history.size())
            # Eqn. 15
            y_tilde = self.fc(torch.cat((context, y_history), dim=1))  # (batch_size, out_size)
            print('y_title:',y_tilde.shape)
            # Eqn. 16: LSTM
            self.lstm_layer.flatten_parameters()
            _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
            hidden = lstm_output[0]  # 1 * batch_size * decoder_hidden_size
            cell = lstm_output[1]  # 1 * batch_size * decoder_hidden_size

        # Eqn. 22: final output
        return self.fc_final(torch.cat((hidden[0], context), dim=1))
        # return F.softmax(x,dim=1)


    def init_hidden(self,x, hidden_size):
        if torch.cuda.is_available():
            return Variable(torch.zeros(1, x.size(0), hidden_size)).cuda()
        else:
            return Variable(torch.zeros(1, x.size(0), hidden_size))