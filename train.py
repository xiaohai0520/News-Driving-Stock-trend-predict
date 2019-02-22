import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dataset import DataSet
from model import Encoder,Decoder
from torch import nn
from torch.autograd import Variable
from torch import optim
import pickle
import torch.nn.functional as F

DRIVING = "D:\Projects\\stock_predict\\stock_data\\Predicting-the-Dow-Jones-with-Headlines-master\\News.csv"
TARGET = "D:\Projects\\stock_predict\\stock_data\\Predicting-the-Dow-Jones-with-Headlines-master\\DowJones.csv"

DECODER_HIDDEN_SIZE = 64
ENCODER_HIDDEN_SIZE = 64
TIME_STEP = 10
SPLIT = 0.8
LR = 0.01
num_epochs = 10
batch_size = 128
interval = 5


class Trainer:

    def __init__(self, driving, target, time_step, split, lr):
        # self.dataset = DataSet(driving, target, time_step, split)


        # f = open('dataset_obj.txt','wb')
        # pickle.dump(self.dataset,f)
        # f.close()

        f = open('dataset_obj.txt','rb')
        self.dataset = pickle.load(f)
        f.close()

        # x,y,y_seq = self.dataset.get_train_set()

        # y =y.long()
        #
        # print(x[0])
        # print(y[0])
        #
        # print(y_seq[0])
        self.encoder = Encoder(input_size=self.dataset.get_num_features(), hidden_size=ENCODER_HIDDEN_SIZE,T=time_step)
        self.decoder = Decoder(encoder_hidden_size=ENCODER_HIDDEN_SIZE,  decoder_hidden_size=DECODER_HIDDEN_SIZE,T=time_step)
        if torch.cuda.is_available():
            # print('tocuda')
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
        self.encoder_optim = optim.Adam(self.encoder.parameters(), lr)
        self.decoder_optim = optim.Adam(self.decoder.parameters(), lr)
        self.loss_func = nn.BCELoss()
        self.train_size, self.test_size = self.dataset.get_size()

    def train_minibatch(self, num_epochs, batch_size, interval):
        x_train, y_train, y_seq_train = self.dataset.get_train_set()
        for epoch in range(num_epochs):
            i = 0
            loss_sum = 0
            while i < self.train_size:
                self.encoder_optim.zero_grad()
                self.decoder_optim.zero_grad()
                batch_end = i + batch_size
                if (batch_end >= self.train_size):
                    batch_end = self.train_size
                var_x = self.to_variable(x_train[i: batch_end])
                var_y = self.to_variable(y_train[i: batch_end])
                var_y_seq = self.to_variable(y_seq_train[i: batch_end])
                if var_x.dim() == 2:
                    var_x = var_x.unsqueeze(2)
                weight1,code = self.encoder(var_x)
                # print('code_size:',code.size())
                y_res = self.decoder(code, var_y_seq)

                # print(y_res[0])
                # print(var_y[0])
                loss = self.loss_func(torch.sigmoid(y_res), var_y)
                loss.backward()
                self.encoder_optim.step()
                self.decoder_optim.step()
                loss_sum += loss.item()
                i = batch_end
            print('epoch [%d] finished, the average loss is %f' % (epoch, loss_sum))
            if (epoch + 1) % (interval) == 0:
                torch.save(self.encoder.state_dict(), 'D:\Projects\\stock_predict\\models\\encoder' + str(epoch + 1) + '-norm' + '.model')
                torch.save(self.decoder.state_dict(), 'D:\Projects\\stock_predict\\models\\decoder' + str(epoch + 1) + '-norm' + '.model')
            if (epoch + 1)% 2 == 0:
                self.test(batch_size)


    def test(self,batch_size):
        x_test, y_test, y_seq_test = self.dataset.get_test_set()
        i = 0
        res = []
        y_labels = []
        for y in y_test:
            if y[0] == 0:
                y_labels.append(1)
            else:
                y_labels.append(0)

        while i < self.test_size:

            batch_end = i + batch_size
            if batch_end >= self.test_size:
                batch_end = self.test_size
            var_x = self.to_variable(x_test[i: batch_end])
            var_y = self.to_variable(y_test[i: batch_end])
            var_y_seq = self.to_variable(y_seq_test[i: batch_end])
            if var_x.dim() == 2:
                var_x = var_x.unsqueeze(2)
            weight1, code = self.encoder(var_x)
            # print('code_size:',code.size())
            y_res = self.decoder(code, var_y_seq)
            y_res = torch.sigmoid(y_res)
            _,predict = torch.max(y_res.data,i)
            res.extend(predict)
            i = batch_end

        res = np.array(res)

        y_labels = np.array(y_labels)


        correct = (res == y_labels).sum()
        print(correct / len(y_test))








    def load_model(self, encoder_path, decoder_path):
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=lambda storage, loc: storage))

    def to_variable(self, x):
        if torch.cuda.is_available():
            # print("var to cuda")
            return Variable(torch.from_numpy(x).float()).cuda()
        else:
            return Variable(torch.from_numpy(x).float())

    def AccuarcyCompute(self,pred, label):
        pred = pred.cpu().data.numpy()
        label = label.cpu().data.numpy()
        correct = (pred == label).sum().item()
        return correct/len(label)



if __name__ == '__main__':


    trainer = Trainer(DRIVING, TARGET, 10, SPLIT, LR)

    trainer.train_minibatch(num_epochs, batch_size, interval)



