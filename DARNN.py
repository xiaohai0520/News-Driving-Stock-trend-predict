import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from lstms import LayerNormGalLSTM

seed = 2019
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class Encoder(nn.Module):


    def __init__(self,input_size, hidden_size, embedding_size,T):

        super(Encoder, self).__init__()

        #25
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.T = T

        self.att1 = nn.Linear(in_features=2 * hidden_size, out_features=self.T)
        self.att2 = nn.Linear(in_features=self.embedding_size, out_features=self.T)

        self.tanh = nn.Tanh()
        self.attn3 = nn.Linear(in_features=self.T, out_features=1)

        self.lstm = nn.LSTM(input_size=embedding_size,hidden_size=hidden_size,batch_first=True)

        self.init_weight_type = 0
        self.init_weight()

    def forward(self,input_data):

        batch_size = input_data.size(0)

        #input data : 128 * 15 * 25 * 768

        input_weight = Variable(torch.zeros(batch_size,self.T,self.input_size))


        #128 * 15 * 128
        input_encoded = self.init_variable(batch_size, self.T, self.hidden_size)


        #1 128 64
        hidden = self.init_variable(1, batch_size, self.hidden_size)
        cell = self.init_variable(1, batch_size, self.hidden_size)

        for t in range(self.T):

            #  x   128 * 25 * (64*2)
            x1 = torch.cat((self.embedding_hidden(hidden),self.embedding_hidden(cell)),dim=2)

            # 128 * 25 * 15
            z1 = self.att1(x1)

            # 128 * 1 * 25 * 768  -> 128 * 25 * 768
            x2 = input_data[:,t,:,:]

            # 128 * 25 * 15
            z2 = self.att2(x2)

            #128 * 25 * 15
            z = z1 + z2

            # 128 * 25 * 1
            z3 = self.attn3(self.tanh(z))
            # Eqn. 9: Softmax the attention weights
            # (batch_size, input_size)  128 * 25
            # if batch_size > 1:
            #     attn_weights = F.softmax(z3.view(batch_size, self.input_size), dim=1)
            # else:
            #     attn_weights = self.init_variable(batch_size, self.input_size) + 1

            # 128 * 1 * 25
            attn_weights = F.softmax(z3.permute(0,2,1),dim=2)

            # 128 * 1 * 768
            weight_input = torch.bmm(attn_weights,x2)

            _,states = self.lstm(weight_input,(hidden,cell))

            hidden = states[0]
            cell = states[1]

            input_encoded[:,t,:] = hidden

        return input_encoded




    def embedding_hidden(self, x):
        return x.repeat(self.input_size, 1, 1).permute(1, 0, 2)


    def init_variable(self, *args):
        zero_tensor = torch.zeros(args)
        if torch.cuda.is_available():
            zero_tensor = zero_tensor.cuda()
        return Variable(zero_tensor)

    def init_weight(self):
        """
        init weight for rnn, bias for all 0 and  ih and hh are init to the different type of distribute
        :return:
        """
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                param.data.fill_(0)
            else:
                if self.init_weight_type == 0:
                    nn.init.normal_(param, 0, 0.5)
                elif self.init_weight_type == 1:
                    nn.init.xavier_normal_(param)
                elif self.init_weight_type == 2:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.uniform_(param)

class Decoder(nn.Module):

    def __init__(self,encoder_hidden_size, decoder_hidden_size, T, incre,drop,out_feats=3):
        super(Decoder, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.T = T
        self.incre = incre
        self.drop = drop

        self.attn1 = nn.Linear(in_features=2*decoder_hidden_size, out_features=encoder_hidden_size)

        self.attn2 = nn.Linear(in_features=encoder_hidden_size, out_features=encoder_hidden_size)

        self.tanh = nn.Tanh()
        self.attn3 = nn.Linear(in_features=encoder_hidden_size, out_features=1)

        self.tilde = nn.Linear(in_features=encoder_hidden_size + 3, out_features=2*decoder_hidden_size)

        self.lstm = nn.LSTM(input_size=2*decoder_hidden_size,hidden_size=decoder_hidden_size,batch_first=True)

        self.fc1 = nn.Linear(in_features=decoder_hidden_size + encoder_hidden_size, out_features=decoder_hidden_size)
        self.fc2 = nn.Linear(in_features=decoder_hidden_size, out_features=out_feats)

        self.dropout = nn.Dropout(p=0.3)
        self.init_weight_type = 0
        self.init_weight()

    def forward(self,input_encoded,y_seq):
        batch_size = input_encoded.size(0)

        # hidden : 1 * batch_size * decode hidden size    1 * 128 * 64
        hidden = self.init_variable(1, batch_size, self.decoder_hidden_size)

        # cell : 1 * batch_size * decode hidden size    1 * 128 * 64
        cell = self.init_variable(1, batch_size, self.decoder_hidden_size)

        # context : batch_size * decode_hidden   128 * 64
        context = self.init_variable(batch_size, self.decoder_hidden_size)

        total_hidden = self.init_variable(batch_size,self.T,2 * self.decoder_hidden_size)

        for t in range(self.T):

            # 128 * T （15）* （2* hidden)
            x1 = torch.cat((self.embedding_hidden(hidden), self.embedding_hidden(cell)), dim=2)

            # 128 * T * 64
            z1 = self.attn1(x1)

            # 128 * T * (encoder_size)
            z2 = self.attn2(input_encoded)

            z = z1 + z2

            # 128 *T * 1
            z3 = self.attn3(self.tanh(z))

            # 128 * 1 * T
            att_weight = F.softmax(z3.permute(0,2,1),dim=2)

            # 128 * 1 * T    128 * T * hidden size   -> 128 * hidden_size
            context = torch.bmm(att_weight, input_encoded).squeeze(1)

            if t < self.T - 1:

                # yc: 128 * (3 + 64) -> 128 * 67
                # print(context.size())
                # print(y_seq.size())
                yc = torch.cat((y_seq[:, t ,:], context), dim=1)



                # (batch_size, out_size)   128 * (2*64)
                y_tilde = self.tilde(yc)

                # Eqn. 16: LSTM
                # self.lstm_layer.flatten_parameters()

                # y_tilde.unsqueeze:  1 * 128 * 1  hidden and cell : 1 * 128 * 64
                _, lstm_state = self.lstm(y_tilde.unsqueeze(1), (hidden, cell))

                # 1 * batch_size * decoder_hidden_size
                hidden = lstm_state[0]
                if self.drop:
                    hidden = self.dropout(hidden)
                # 1 * batch_size * decoder_hidden_size
                cell = lstm_state[1]

            total_hidden[:,t,:] = torch.cat((hidden.squeeze(0),context),dim=1)
        if self.incre:
            return self.fc2(self.fc1(total_hidden))
        else:
            return self.fc2(self.fc1(torch.cat((hidden.squeeze(0), context), dim=1)))


    def init_variable(self, *args):
        zero_tensor = torch.zeros(args)
        if torch.cuda.is_available():
            zero_tensor = zero_tensor.cuda()
        return Variable(zero_tensor)

    def embedding_hidden(self, x):
        return x.repeat(self.T, 1, 1).permute(1, 0, 2)

    def init_weight(self):
        """
        init weight for rnn, bias for all 0 and  ih and hh are init to the different type of distribute
        :return:
        """
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                param.data.fill_(0)
            else:
                if self.init_weight_type == 0:
                    nn.init.normal_(param, 0, 0.5)
                elif self.init_weight_type == 1:
                    nn.init.xavier_normal_(param)
                elif self.init_weight_type == 2:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.uniform_(param)

#
# e = Encoder(25,128,768,15)
# d = Decoder(128,128,15)
# e = e.cuda()
# d = d.cuda()
# input = torch.rand(128,15,25,768)
# input = input.cuda()
# y_seq = torch.rand(128,15,3)
# y_seq = y_seq.cuda()
# code = e(input)
# out = d(code,y_seq)
# print(out.size())