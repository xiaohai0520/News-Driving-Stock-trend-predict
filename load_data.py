import re
import gensim
import pandas as pd
import pickle
import numpy as np
# import torch
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors

from nltk.corpus import stopwords

# global file path
file = "D:\\Projects\\stock_predict\\glove.6B\\glove.6B.50d.txt"
glove_vector_name = "gensim_glove_vectors.txt"
news_file = "D:\Projects\\stock_predict\\stock_data\\Predicting-the-Dow-Jones-with-Headlines-master\\News.csv"
price_file = "D:\Projects\\stock_predict\\stock_data\\Predicting-the-Dow-Jones-with-Headlines-master\\DowJones.csv"
embedding_dim = 50

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







def transfer_model(input,output):
    """

    :param file: file path of the glove txt file
    :return: None
    """
    glove2word2vec(glove_input_file=input, word2vec_output_file=output)

def load_glove_model(input):
    """

    :param filename: the file name
    :return: glove model
    """
    return KeyedVectors.load_word2vec_format(input, binary=False)


def read_news():
    dj = pd.read_csv(price_file)
    news = pd.read_csv(news_file)
    return dj,news

"""
dj,news = read_news()
print(dj.isnull().sum())
print(news.isnull().sum())


print(dj.shape)
print(news.shape)


# Compare the number of unique dates. We want matching values.
print(len(set(dj.Date)))
print(len(set(news.Date)))


# Remove the extra dates that are in news
news = news[news.Date.isin(dj.Date)]

print(len(set(dj.Date)))
print(len(set(news.Date)))

#use the date as index to get the different each day
dj = dj.set_index('Date').diff(periods=1)
print(dj.head())
#add the date back
dj['Date'] = dj.index

dj = dj.reset_index(drop=True)
# Remove unneeded features
dj = dj.drop(['High','Low','Close','Volume','Adj Close'], 1)
print(dj.head())

#remove the first row
dj = dj[dj.Open.notnull()]

prices_trend = []
headlines = []
for row in dj.iterrows():
    # print(row)
    # print('row0',row[0])
    daily_headlines = []
    # print('row1',row[1])
    date = row[1]['Date']
    prices_trend.append(1 if row[1]['Open']>= 0 else -1)
    for row_ in news[news.Date == date].iterrows():
        daily_headlines.append(row_[1]['News'])
    headlines.append(daily_headlines)



# print(prices_trend)
# print(headlines)
# print(len(headlines))
# print(len(prices_trend))

def clean_text(text, remove_stopwords=True):
    '''Remove unwanted characters and format the text to create fewer nulls word embeddings'''

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

        # stopwords = nltk.download('stopwords')
        # print(stopwords)
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    return text

clean_headlines = []

for daily_headlines in headlines:
    clean_daily_headlines = []
    for headline in daily_headlines:
        clean_daily_headlines.append(clean_text(headline))
    clean_headlines.append(clean_daily_headlines)


#clean_headlines is the set of dates news
print(clean_headlines[0])
news1 = clean_headlines[0]
f = open('news1.txt','wb')
pickle.dump(news1,f)

f = open('news1.txt','rb')
anews1 = pickle.load(f)

print(anews1)

"""


def dailynews_to_vector():
    """

    :return: daily news numpy array
    """
    f = open('news1.txt','rb')
    news_corpos = pickle.load(f)
    # print(news)

    news_vectors = np.empty(len(news_corpos))
    for news in news_corpos:
        ls = news.split()
        length = len(ls)

        single_news_vec = np.zeros(50)
        for word in ls:
            print(word)
            try:
                single_news_vec += np.asarray(model[word])
            except KeyError:
                print('No ')
                single_news_vec += np.random.uniform(-1.0, 1.0, embedding_dim)
            # if model[word] is not None:
            #     single_news_vec += np.asarray(model[word])
            # else:
            #     single_news_vec += np.random.uniform(-1.0, 1.0, embedding_dim)
            # print(single_news_vec)
        single_news_vec /= length
        print(single_news_vec)
        news_vectors.append(single_news_vec)
    print(len(news_corpos))
    print(len(news_vectors))



# transfer_model(file,glove_vector_name)
model = load_glove_model(glove_vector_name)
dailynews_to_vector()