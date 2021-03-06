import re
# import gensim
import pandas as pd
import pickle
import numpy as np
import nltk
# from gensim.scripts.glove2word2vec import glove2word2vec
# from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import stopwords
from bert_serving.client import BertClient

DRIVING = ".\\stock_data\\News.csv"
TARGET = ".\\stock_data\\DowJones.csv"

# global file path
# file = "D:\\Projects\\stock_predict\\glove.6B\\glove.6B.50d.txt"
# glove_vector_name = "gensim_glove_vectors.txt"
embedding_dim = 768

# A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are"
}


class DataSet():

    def __init__(self,newsfile,pricesfile,time_step,split_ratio=0.6):

        #the news file path
        self.newsfile = newsfile

        #the price file path
        self.pricesfile = pricesfile

        #create a BERT Client object to do the sentence embedding
        # self.bc = BertClient(check_length=False)

        #transfer the txt into w2v file
        # self.transfer_model(file,glove_vector_name)

        #get the word2vec model
        # self.model = self.load_glove_model(glove_vector_name)


        #read from the csv
        prices_trend, headlines = self.get_all_data()

        self.news, self.y = self.create_dataset(prices_trend, headlines, time_step)
        #
        # print(self.news.shape)
        #
        self.train_size = int(split_ratio * (len(self.y) - time_step - 1))

        self.validation_size = (len(self.y) - self.train_size)//2

        self.test_size = len(self.y) - self.train_size - self.validation_size

    def get_size(self):
        return self.train_size, self.validation_size, self.test_size

    def get_num_features(self):
        return self.news[0].shape[1]

    def get_train_set(self):
        return self.news[:self.train_size], self.y[:self.train_size]

    def get_validation_set(self):
        return self.news[self.train_size:self.train_size+self.validation_size],self.y[self.train_size:self.train_size+self.validation_size]

    def get_test_set(self):
        return self.news[self.train_size+self.validation_size:],self.y[self.train_size+self.validation_size:]



    # def transfer_model(self,input, output):
    #     """
    #     transfer txt into model with glove lib
    #     :param file: file path of the glove txt file
    #     :return: None
    #     """
    #     glove2word2vec(glove_input_file=input, word2vec_output_file=output)


    # def load_glove_model(self,input):
    #     """
    #     get the glove model
    #     :param filename: the file name
    #     :return: glove model
    #     """
    #     return KeyedVectors.load_word2vec_format(input, binary=False)

    def read_news(self):
        """
        read the data from the csv
        :return: both of the news and stock price
        """
        prices = pd.read_csv(self.pricesfile)
        news = pd.read_csv(self.newsfile)
        return prices, news


    def clean_text(self,text, remove_stopwords=True):
        """
        Remove unwanted characters and format the text to create fewer nulls word embeddings
        :param text: text
        :param remove_stopwords: boolean to remove the stopword
        :return: clleaned news
        """


        # Convert words to lower case
        text = text.lower()

        # Replace contractions with their longer forms
        if True:
            text = text.split()
            new_text = []
            for word in text:
                if word in contractions:
                    new_text.append(contractions[word])
                else:
                    new_text.append(word)
            text = " ".join(new_text)

        # Format words and remove unwanted characters
        text = re.sub(r'&amp;', '', text)
        text = re.sub(r'0,0', '00', text)
        text = re.sub(r'[_"\-;%()|.,+&=*%.,!?:#@\[\]]', ' ', text)
        text = re.sub(r'\'', ' ', text)
        text = re.sub(r'\$', ' $ ', text)
        text = re.sub(r'u s ', ' united states ', text)
        text = re.sub(r'u n ', ' united nations ', text)
        text = re.sub(r'u k ', ' united kingdom ', text)
        text = re.sub(r'j k ', ' jk ', text)
        text = re.sub(r' s ', ' ', text)
        text = re.sub(r' yr ', ' year ', text)
        text = re.sub(r' l g b t ', ' lgbt ', text)
        text = re.sub(r'0km ', '0 km ', text)

        # Optionally, remove stop words
        if remove_stopwords:
            text = text.split()

            #         nltk.download('stopwords')
            #         print(stopwords)
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]
            text = " ".join(text)

        return text

    def get_data_from_csv(self):
        """
        read data from csv and return the array
        :return: stock trend and news array
        """
        prices, news = self.read_news()

        # search the null length
        # print(dj.isnull().sum())
        # print(news.isnull().sum())

        # looke the shape
        # print(dj.shape)
        # print(news.shape)

        # Compare the number of unique dates. We want matching values.
        # print(len(set(dj.Date)))
        # print(len(set(news.Date)))

        # Remove the extra dates that are in news
        news = news[news.Date.isin(prices.Date)]

        # make sure the equal
        # print(len(set(dj.Date)))
        # print(len(set(news.Date)))


        # remove un need features
        prices = prices.drop(['High', 'Low', 'Close', 'Volume', 'Adj Close'], 1)

        # prices = prices[::-1]
        prices = prices[::-1].reset_index(drop=True)

        # print(prices.head())



        prices_diff = prices['Open'].reset_index(drop=True).diff()



        prices_diff.index = prices_diff.index - 1
        # print(prices_diff.head())
        # print(prices_open[1:].reset_index(drop=True).head())
        # print('prices_open',prices_open.head())

        # prices = prices.set_index('Date').diff()
        # prices['Date'] = prices.index
        # prices = prices.reset_index(drop=True)
        # print(prices.head())



        prices['Open_Diff'] = prices_diff






        # remove the first row
        prices = prices[prices.Open.notnull()]

        # print(prices.head())
        # two array to save the trend(label) and news(input_string)
        prices_trend = []
        headlines = []
        for row in prices.iterrows():
            # print(row)
            # print('row0',row[0])
            daily_headlines = []
            # print('row1',row[1])
            date = row[1]['Date']


            """ 0: down    1: preserve      2:up"""

            if row[1]['Open_Diff']/row[1]['Open'] < -0.0041:
                prices_trend.append(0)
            elif -0.0041 <= row[1]['Open_Diff']/row[1]['Open'] < 0.0087:
                prices_trend.append(1)
            else:
                prices_trend.append(2)




            for row_ in news[news.Date == date].iterrows():
                daily_headlines.append(row_[1]['News'])
            headlines.append(daily_headlines)


        return prices_trend, headlines

    def clean_new(self,headlines):
        """
        clean the whole news data
        :param headlines: the news data set
        :return: cleaned news data set
        """
        clean_headlines = []

        for daily_headlines in headlines:
            clean_daily_headlines = []
            for headline in daily_headlines:
                clean_daily_headlines.append(self.clean_text(headline))
            clean_headlines.append(clean_daily_headlines)
        return clean_headlines


    # def dailynews_to_vector(self,news_corpos):
    #     """
    #
    #     :return: daily news numpy array
    #     """
    #
    #     #     news_vectors = np.empty(len(news_corpos))
    #     news_vectors = []
    #     for news in news_corpos:
    #         ls = news.split()
    #         length = len(ls)
    #         single_news_vec = []
    #         for word in ls:
    #             #             print(word)
    #             try:
    #                 single_news_vec.append(np.asarray(self.model[word]))
    #             except KeyError:
    #                 #                 print('No word')
    #                 single_news_vec.append(np.random.uniform(-1.0, 1.0, embedding_dim))
    #         #size 22 * 50
    #         single_news_vec = np.array(single_news_vec)
    #         # size n * 50
    #
    #         # print('single news vector:',single_news_vec.shape)
    #
    #         #
    #         news_vectors.append(single_news_vec.mean(0))
    #     # 22 * 50
    #     news_vectors = np.array(news_vectors)
    #     # print('daily total news vecotr:',news_vectors.shape)
    #     return news_vectors



    # def pooling(self,dailynews):
    #     news_max = dailynews.max(0)
    #     news_min = dailynews.min(0)
    #     news_mean = dailynews.mean(0)
    #
    #     dailynews = np.concatenate((news_max,news_min,news_mean),axis=None)
    #     return dailynews

    # def news_embedding(self,headlines):
    #     headlines_vector = []
    #     for dailynews in headlines:
    #         day_news = self.dailynews_to_vector(dailynews)
    #
    #         headlines_vector.append(day_news)
    #
    #     return np.array(headlines_vector)


    def news_embedding(self,headlines):
        """
        do the news embedding op
        :param headlines: the whole news data set
        :return: the data set of embedding news
        """
        res = []
        for headline in headlines:
            # print(headline)

            hd_embedding_array = bc.encode(headline)

            # print(hd_embedding_array)
            res.append(hd_embedding_array)
        return np.array(res)


    def create_dataset(self,prices_trend, headlines, time_step):
        """

        :param prices_trend: stock list
        :param headlines: news list
        :param time_step: time step 15
        :return: three np array
        """
        x, y = [], []
        for i in range(len(prices_trend) - time_step - 1):
            last = i + time_step
            x.append(headlines[i:last])

            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            y.append(prices_trend[last])


        return np.array(x), np.array(y)

    def get_all_data(self):
        """

        :return: the stock predict(list) and news np array
        """

        #get the original data of trend and headline
        prices_trend, headlines = self.get_data_from_csv()

        # print(prices_trend)
        # print(type(prices_trend))

        #do the word cleaning

        #stylle:[[news,news,news],
        #        [news,news,news]
        #        [news,news,news]]
        headlines = self.clean_new(headlines)

        # print(headlines)

        # for headline in headlines:
        #
        #     print(headline)


        #do the news embedding

        headlines = self.news_embedding(headlines)

        # print(headlines[:5])

        #
        min_length = min(len(i) for i in headlines)
        max_length = max(len(i) for i in headlines)
        cur_headlines = []

        #make full to 25
        for headline in headlines:
            if len(headline) == max_length:
                cur_headlines.append(headline)
            else:
                cur_headlines.append(np.concatenate((headline,np.zeros((max_length - len(headline),embedding_dim)))))


        # prices type is list
        return prices_trend, np.array(cur_headlines)

#
# bc = BertClient(check_length=False)
# # #
# ds = DataSet(DRIVING, TARGET,15)
#
# f = open('dataset_obj.txt','wb')
# pickle.dump(ds,f,protocol=4)
# f.close()
