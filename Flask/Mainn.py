#!/usr/bin/env python
# coding: utf-8

# In[66]:


# import import_ipynb

import pickle


# In[2]:


import pandas as pd
import numpy as np
import re


# In[3]:


from sklearn import svm


# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


# In[6]:


#import Pathconfig as pg


# In[7]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# In[8]:


import enchant
from textblob import Word
from textblob import TextBlob


# In[9]:


from nltk.stem import PorterStemmer
ps=PorterStemmer()


# In[10]:


from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()


# In[11]:


from sklearn.metrics import accuracy_score, f1_score


# In[12]:


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor


# In[13]:


from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    recall_score,
    precision_score,
)

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import make_scorer, accuracy_score, f1_score


# In[14]:


#paths
pathdata='C:/Users/su/Desktop/Major_Project_work/data/data.csv'
pathStopwords = 'C:/Users/su/Desktop/Major_Project_work/Models/stopwords.txt'


# In[15]:


open(pathStopwords)


# In[16]:


#extract dataset and removing undefined , replace PI and No PI with yes and no.
def extract(path):
    fd = open(path, encoding="utf-8", errors='replace')
    df = pd.read_csv(fd)
    defined = df['class'] != ("undefined")
    # #output dataframe without undeined
    df2 = df[defined]
    defined1 = df2['class'] != "Undefined"
    df4 = df2[defined1]
    # replace no PI with no
    df3 = df4.replace("No PI", "no")
    # replace PI with yes
    final = df3.replace("PI", "yes")
    replace_yes = final.replace("Yes", "yes")
    final_df = replace_yes.replace("No", "no")
    return final_df, df


# In[17]:


final_data_frame , data_frame_undefined =extract(pathdata)


# In[18]:


data_frame_undefined


# In[19]:


# LOWERCASE
final_data_frame["text"] = final_data_frame["text"].apply(
    lambda x: " ".join(x.lower() for x in x.split())
)
print("lowercase all text")
print(final_data_frame["text"].head())
print()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


y = (data_frame_undefined['class']) .value_counts()
print(y)


# In[21]:


yes_no_count = (final_data_frame['class']) .value_counts()
print(y)


# In[22]:


def read_stopwords(path):
    file1 = open(path, "r")
    stopword = file1.readlines()
    file1.close()
    li_stopwords = stopword[0].split()
    return li_stopwords


# In[ ]:





# In[23]:


#check spelling 
def check_english(temp_df):
    d = enchant.Dict("en_US")
    #print(d)
    new_eng = pd.DataFrame()
    count = 0

    for sentence in temp_df['text']:
        temp_sent = ""
        for word in sentence.split():
            temp = word.lower()
            if d.check(word):
               # print("ammar")
                temp_sent = temp_sent + temp + " "
        #print(temp_sent)
        new_eng.at[count, 'text'] = temp_sent
        new_eng.at[count, 'class'] = temp_df.iloc[count]['class']
        count += 1
    #print(new_eng)
    return new_eng


# In[ ]:





# In[ ]:





# In[24]:


#new_eng = check_english(final_data_frame)
new_eng = final_data_frame

# In[25]:


def Stemming(temp_df):
    new_eng = pd.DataFrame()
    count = 0
    
    for sentence in temp_df['text']:
        temp_sent = ""
        for word in sentence.split():
            temp = ps.stem(word.lower())
            temp_sent = temp_sent + temp + " "
        print(temp_sent)
        new_eng.at[count, 'text'] = temp_sent
        new_eng.at[count, 'class'] = temp_df.iloc[count]['class']
        count += 1
    # print(new_eng)
    return new_eng


# In[26]:


def lemmatizing(temp_df):
    new_eng = pd.DataFrame()
    count = 0
    
    for sentence in temp_df['text']:
        temp_sent = ""
        for word in sentence.split():
            temp = wordnet_lemmatizer.lemmatize(word.lower())
            temp_sent = temp_sent + temp + " "
        #print(temp_sent)
        new_eng.at[count, 'text'] = temp_sent
        new_eng.at[count, 'class'] = temp_df.iloc[count]['class']
        count += 1
    # print(new_eng)
    return new_eng


# In[27]:


stem_df=lemmatizing(new_eng)


# In[28]:


def space(final_df):
    new_df = pd.DataFrame()
    count_tweets = 0
    for text in final_df['text']:
        temp = ""
        for char in text:
            if char in [",", ".", "!", "?", ":", ";"]:
                temp = temp + ' ' + char
                
            else:
                temp = temp + char
        # print(temp)
        new_df.at[count_tweets, 'text'] = temp
        new_df.at[count_tweets,'class'] = final_df.iloc[count_tweets]['class']
        count_tweets += 1
    # print("new_df")
    # print(new_df)
    return new_df


# In[29]:


remove_space = space(stem_df)
print(remove_space)


# In[30]:


def remove_punc(temp_df):
    count = 0
    for text in temp_df['text']:
        out = re.sub(r'[^\w\s]', '', text)
        temp_df.at[count, 'text'] = out
        temp_df.at[count, 'class'] = temp_df.iloc[count]['class']
        count += 1
    return temp_df


# In[31]:


remove_punch_df= remove_punc(remove_space)
#print(remove_punch_df)


# In[32]:


def remove_stopwords(df_punc_remove):
    stop_words = set(stopwords.words('english'))
    li_stopwords = read_stopwords(pathStopwords)
    #print(stop_words)
    count_clean = 0
    for text in df_punc_remove['text']:
        word_tokens = word_tokenize(text)
        clean_text = ""
        for w in word_tokens:
            if w.lower() not in li_stopwords:
                clean_text = clean_text + w.lower() + ' '
        df_punc_remove.at[count_clean, 'text'] = clean_text
        df_punc_remove.at[count_clean,'class'] = df_punc_remove.iloc[count_clean]['class']
        count_clean += 1
    # return list of corpus without stop words in a list.
    #print(df_punc_remove)
    return df_punc_remove


# In[33]:


remove_stopwords(remove_punch_df)


# In[34]:


def text_concat(final_df):
    text = ""
    for x in final_df["text"]:
        text = text + str(x)
    return text


# In[35]:


def clean_data(final_df):
    # print(final_df)
    #eng_df =check_english(final_df)
    eng_df = final_df
    # print(eng_df)
    df_stem =lemmatizing(eng_df)
    new_df =space(eng_df)
    remove_punc_df = remove_punc(new_df)
    cleaned_text =remove_stopwords(remove_punc_df)
    # print(cleaned_text)
    return cleaned_text


# In[36]:


clean_text = clean_data(final_data_frame)


# In[37]:


print(clean_text)


# In[38]:


# BUILDING THE CORPUS
corpus = []
for text in clean_text["text"]:
    corpus.append(text)
   # print(text)
# print(corpus)


# In[39]:


# CHANGE CLASS VALUES FROM YES/NO TO 0/1
final_data_frame.rename(columns={"class": "class_label"}, inplace=True)
Class_Label = {"yes": 1, "no": 0}
final_data_frame.class_label = [Class_Label[item] for item in final_data_frame.class_label]
final_data_frame.rename(columns={"class_label": "class"}, inplace=True)
print("rename values of class column")
print(final_data_frame.head())
print()
# ------


# In[40]:


# IDF
# Performs the TF-IDF transformation from a provided matrix of counts.

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorized_data = tfidf_vectorizer.fit_transform(corpus)


# In[41]:


print(tfidf_vectorized_data)


# In[42]:


# chose document vector
vectorized_data = tfidf_vectorized_data


# In[43]:


vectorized_data.shape


# In[44]:


# SPLITING THE DATA
X_train, X_test, Y_train, Y_test = train_test_split(vectorized_data, final_data_frame["class"], test_size=0.3, random_state=0)


# In[45]:


X_test.shape


# In[46]:


Y_test.shape


# In[47]:


SVM = svm.SVC(probability=True, C=0.75, kernel="linear", degree=3)
SVM.fit(X_train, Y_train)
    


# In[48]:


# create regressor object
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train,Y_train)


# In[49]:


pred = regressor.predict(X_test)
print(pred)


# In[50]:


def report_results(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)

    auc = roc_auc_score(y, pred_proba)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    TrueNeg = tn / (tn + fp)
    result = {
        "auc": auc,
        "f1": f1,
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "True Negative rate": TrueNeg,
    }
    return result


# In[51]:


X_test.shape


# In[52]:


# statitics for SVM
stats = report_results(SVM, X_train, Y_train)
print("-------------------------------------------------------------------------")
print("statitics for SVM")
print(stats)
print("-------------------------------------------------------------------------")
print()


# In[ ]:





# In[ ]:





# In[ ]:





# In[53]:


print(X_train)


# In[54]:


# importing random forest classifier from assemble module
from sklearn.ensemble import RandomForestClassifier


# In[55]:


#creating a RF classifier
RFC = RandomForestClassifier(n_estimators = 100)  


# In[56]:


# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
RFC.fit(X_train, Y_train)


# In[57]:


# performing predictions on the test dataset
y_pred = RFC.predict(X_train)


# In[58]:


# metrics are used to find accuracy or error
from sklearn import metrics  


# In[ ]:





# In[59]:


# statitics for RFC
stats_for_RFC = report_results(RFC, X_test, Y_test)
print("-------------------------------------------------------------------------")
print("statitics for SVM")
print(stats)
print("-------------------------------------------------------------------------")
print()

stats_for_SVM = report_results(SVM,X_test,Y_test)


# In[ ]:





# In[60]:


list1=['ill bye iphone','iphone is all bad that they are all the same']
tfidf_vectorized_data1 = tfidf_vectorizer.transform(list1)
print(tfidf_vectorized_data1)


# In[61]:


tfidf_vectorized_data1.shape


# In[62]:


y_pred = SVM.predict(X_train)
y_pred.shape


# In[63]:


yhat = RFC.predict(tfidf_vectorized_data1)
print(yhat)


# In[64]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean = False)


# In[65]:


# # Train an SVM classifier on the dataset
# svm_clf = SVC(kernel="linear", C=0.75)

# svm_clf.fit(X_train, Y_train)


# # Plot the dataset and the decision boundary
# x0s = np.linspace(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1, 1000)
# x1s = np.linspace(X_train[:, 1].min() - 1, X_train[:, 1].max() + 1, 1000)
# x0, x1 = np.meshgrid(x0s, x1s)
# #X_new = np.c_[x0.ravel(), x1.ravel()]
# # y_pred = svm_clf.predict(X_new).reshape(x0.shape)
# y_pred.shape
# #y_pred = y_pred.reshape(x0.shape)
# #plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
# # plt.scatter(X_train[:, 0], X_train[:, 1], c=1, cmap=plt.cm.brg, s=30)
# # plt.xlabel("Petal length")
# # plt.ylabel("Petal width")
# # plt.xlim(x0.min(), x0.max())
# # plt.ylim(x1.min(), x1.max())
# # plt.show()


# In[68]:


pickle.dump(SVM , open('model.pkl','wb'))


# 

# In[69]:


model=pickle.load(open('model.pkl','rb'))


# In[71]:


model.predict(tfidf_vectorized_data1)


# In[ ]:




