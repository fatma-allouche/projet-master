# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 13:49:03 2020

@author: Rahma
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 00:29:33 2020

@author: Rahma
"""


import pandas as pd
import string 
#import nltk
from stop_words import get_stop_words
#from nltk.corpus import stopwords

dataset = pd.read_excel('nadiF.xlsx',encoding='utf-8')
dataset['text'] = dataset['text'].str.lower()
dev = pd.read_excel('devF.xlsx',encoding='utf-8')
dev['text'] = dev['text'].str.lower()
#dev = pd.read_excel("dev.xlsx",encoding='utf-8')
#dataset = dataset[dataset.country_label != 'MSA']
#dataset=dataset.to_csv('train.csv',encoding='utf-8')
#ponctuation 
def remove_punctuation(text):
    no_punct="".join([c for c in text if c not in string.punctuation])
    return no_punct
dataset['text']=dataset.text.apply(str)
dev['text']=dev.text.apply(str)

dataset['text']=dataset['text'].apply (lambda x : remove_punctuation (x))
dev['text']=dev['text'].apply (lambda x : remove_punctuation (x))

# stpword standard 
"""
def remove_stopwords(text):
    no_stop=" ".join([c for c in text.split() if c not in stopwords.words('arabic')])
    return no_stop
dataset['text']=dataset.text.apply(str)
dataset['text']=dataset['text'].apply(lambda x : remove_stopwords(x))
"""
caractere_speciaux=['؟','#','،','«','»','‹','›','„','‚','…','!','¡','?']
def carac(text):
    no_caracs="".join([c for c in text if c not in caractere_speciaux])
    return no_caracs
dataset['text']=dataset['text'].apply (lambda x : carac(x))
dev['text']=dev['text'].apply (lambda x : carac(x))

#dev['text']=dev['text'].apply (lambda x : carac(x))

def remove_http(text):
    no_http=" ".join([c for c in text.split() if c.startswith('http')==False])
    return no_http
dataset['text']=dataset['text'].apply (lambda x : remove_http(x))
dev['text']=dev['text'].apply (lambda x : remove_http(x))

def remove_pic(text):
    no_pic=" ".join([c for c in text.split() if c.startswith('pictwitter')==False])
    return no_pic
dataset['text']=dataset['text'].apply (lambda x : remove_pic(x))
dev['text']=dev['text'].apply (lambda x : remove_pic(x))


#dev['text']=dev['text'].apply (lambda x : remove_pic(x))

def remove_tag(text):
    no_tag=" ".join([c for c in text.split() if c.startswith('@')==False])
    return no_tag
dataset['text']=dataset['text'].apply (lambda x : remove_tag(x))
dev['text']=dev['text'].apply (lambda x : remove_tag(x))

chiffre=['0','1','2','3','4','5','6','7','8','9']
def chiff(text):
    no_chiffre="".join([c for c in text if c not in chiffre])
    return no_chiffre
# la nouvelle partie 
dataset['text']=dataset['text'].apply (lambda x : chiff(x))
dev['text']=dev['text'].apply (lambda x : chiff(x))

#dataset['text'] =dataset['text'].str.replace(' ','#')
#dev['text'] =dev['text'].str.replace(' ','#')
dataset=dataset.dropna(axis=1, thresh=2)
dev=dev.dropna(axis=1, thresh=2)


from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1,1), use_idf=False)



#X = tv.fit_transform(dataset['text'])
"""
X_train = dataset.loc[:106000, 'text'].values
y_train = dataset.loc[:106000, 'country_label'].values
X_test = dataset.loc[106001:, 'text'].values
y_test = dataset.loc[106001:, 'country_label'].values
train = tv.fit_transform(X_train)
test = tv.transform(X_test)
"""
X_train = dataset.loc[:, 'text'].values
y_train = dataset.loc[:, 'country_label'].values
X_test = dev.loc[:, 'text'].values
y_test = dev.loc[:, 'country_label'].values
train = tv.fit_transform(X_train)
test = tv.transform(X_test)
# fin de la nouvelle partie 

#╚feature_cols = tv.get_feature_names()
#from sklearn.model_selection import train_test_split


#X_train, X_test, y_train, y_test=train_test_split(X,dataset['country_label'],random_state=101,test_size=0.3)
print ('----------------------SVM------------------------')


from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(train, y_train)

#result = loaded_model.score(X_test, Y_test)
#print(result)
# some time later...
 
# load the model from disk

from  sklearn.metrics  import accuracy_score
y_pred = svclassifier.predict(test)
print(accuracy_score(y_test,y_pred))
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print ('Accuracy Score :',accuracy_score(y_test, y_pred) )

#file1.close()



