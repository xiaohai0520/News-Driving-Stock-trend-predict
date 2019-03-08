import torch
import matplotlib
# matplotlib.use('Agg')
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
SPLIT = 0.6
LR = 0.001
momentum = 0.9
num_epochs = 50
batch_size = 128
interval = 5
# best_dev_acc = 0.0

class Trainer:

    def __init__(self, driving, target, time_step, split,lr):
        self.dataset = DataSet(driving, target, time_step, split)


        f = open('dataset_obj.txt','wb')
        pickle.dump(self.dataset,f)
        f.close()

        print('save model finish!!!!!!!!!!!!!!!!!!')

        # f = open('dataset_obj.txt','rb')
        # self.dataset = pickle.load(f)
        # f.close()

        self.encoder = Encoder(input_size=self.dataset.get_num_features(), hidden_size=ENCODER_HIDDEN_SIZE,T=time_step)
        self.decoder = Decoder(encoder_hidden_size=ENCODER_HIDDEN_SIZE,  decoder_hidden_size=DECODER_HIDDEN_SIZE,T=time_step)
        if torch.cuda.is_available():
            # print('tocuda')
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
        self.encoder_optim = optim.Adam(self.encoder.parameters(), lr)
        self.decoder_optim = optim.Adam(self.decoder.parameters(), lr)
        self.loss_func = nn.CrossEntropyLoss()
        self.train_size, self.validation_size,self.test_size = self.dataset.get_size()
        self.best_dev_acc = 0.0

    def get_accuracy(self,truth,pred):
        assert len(truth) == len(pred)
        right = (truth == pred).sum()
        return right/len(truth)

    def train_minibatch(self, num_epochs, batch_size, interval):
        train_acc_list = []
        dev_acc_list = []
        train_loss_list = []
        dev_loss_list = []
        x_train, y_train, y_seq_train = self.dataset.get_train_set()
        # print(x_train.shape)
        for epoch in range(num_epochs):
            print('Start epoch {}'.format(epoch))
            i = 0
            loss_sum = 0
            pred_res_total = []
            while i < self.train_size:
                self.encoder_optim.zero_grad()
                self.decoder_optim.zero_grad()
                batch_end = i + batch_size
                if (batch_end >= self.train_size):
                    batch_end = self.train_size
                var_x = self.to_variable(x_train[i: batch_end])
                # var_y = self.to_variable(y_train[i: batch_end])
                var_y = Variable(torch.from_numpy(y_train[i: batch_end]).long()).cuda()
                var_y_seq = self.to_variable(y_seq_train[i: batch_end])
                if var_x.dim() == 2:
                    var_x = var_x.unsqueeze(2)
                code = self.encoder(var_x)

                y_res = self.decoder(code, var_y_seq)

                loss = self.loss_func(y_res, var_y)
                if i == 0:
                    print("y_res:",y_res)
                    print("var_y:",var_y)
                loss.backward()
                self.encoder_optim.step()
                self.decoder_optim.step()
                loss_sum += loss.item()

                # update the i
                i = batch_end

                pred_y = y_res.data.cpu()

                # print('see what the pred and truth')
                # print('y_res:',y_res.shape,' : ',y_res)
                # print('var_y:',var_y.shape,' : ',var_y)

                pred_y = torch.max(F.softmax(pred_y,dim=1), 1)[1]
                #
                # print('pred_y:',pred_y)
                # print('var_y',var_y)

                pred_res_total.extend(pred_y)



                # if i%50 == 0:
                #     print('         finish {0:.2f}/100'.format(i/self.train_size))

            acc = self.get_accuracy(y_train,np.array(pred_res_total))
            print('epoch [%d] finished, the average loss is %.2f, accuracy is %.1f' % (epoch, loss_sum ,acc*100))

            dev_acc,dev_loss = self.test(batch_size)
            print('dev_acc is %.2f'% (dev_acc * 100))

            train_acc_list.append(acc)
            dev_acc_list.append(dev_acc)
            train_loss_list.append(loss_sum)
            dev_loss_list.append(dev_loss)

            if dev_acc > self.best_dev_acc:

                torch.save(self.encoder.state_dict(), 'D:\Projects\\stock_predict\\models\\encoder_best.model')
                torch.save(self.decoder.state_dict(), 'D:\Projects\\stock_predict\\models\\decoder_best.model')
                self.best_dev_acc = dev_acc

                test_acc,test_loss = self.test(batch_size,True)
                print('test_accuracy: %.1f' %(test_acc*100))

        return train_acc_list, dev_acc_list,train_loss_list,dev_loss_list



    def test(self,batch_size,is_test=False):
        if not is_test:
            x, y, y_seq = self.dataset.get_validation_set()
        else:
            x, y, y_seq = self.dataset.get_test_set()
        i = 0
        res = []
        length = len(y)
        loss_sum = 0
        while i < length:

            batch_end = i + batch_size
            if batch_end >= length:
                batch_end = length
            var_x = self.to_variable(x[i: batch_end])
            var_y = Variable(torch.from_numpy(y[i: batch_end]).long()).cuda()
            # var_y = self.to_variable(y_test[i: batch_end])
            var_y_seq = self.to_variable(y_seq[i: batch_end])
            if var_x.dim() == 2:
                var_x = var_x.unsqueeze(2)

            # to encoder get encoder output
            code = self.encoder(var_x)

            # to decoder get classification
            y_res = self.decoder(code, var_y_seq)

            loss = self.loss_func(y_res, var_y)
            loss_sum += loss.item()

            pred_y = y_res.data.cpu()
            pred_y = torch.max(pred_y, 1)[1]
            res.extend(pred_y)
            i = batch_end

        res = np.array(res)

        return self.get_accuracy(y,res),loss_sum



    def load_model(self, encoder_path, decoder_path):
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=lambda storage, loc: storage))

    def to_variable(self, x):
        if torch.cuda.is_available():
            # print("var to cuda")
            return Variable(torch.from_numpy(x).float()).cuda()
        else:
            return Variable(torch.from_numpy(x).float())

    def draw_plot(self,train_list,dev_list,acc=True):
        plt.plot(np.array(train_list))
        plt.plot(np.array(dev_list))
        if acc:
            plt.title('model acc')
            plt.ylabel('accuracy')

        else:
            plt.title('model loss')
            plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','validation'],loc = 'upper left')
        plt.show()




if __name__ == '__main__':


    trainer = Trainer(DRIVING, TARGET, 10, SPLIT, LR)

    train_acc_list, dev_acc_list,train_loss_list,dev_loss_list = trainer.train_minibatch(num_epochs, batch_size, interval)

    trainer.draw_plot(train_loss_list, dev_loss_list,False)
    trainer.draw_plot(train_acc_list, dev_acc_list,True)


