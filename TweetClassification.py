#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 15:51:43 2018

@author: singh
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import math
import string
from collections import Counter
import time

start = time.time()

#Func to find the count of a word in a list
#Probability is calculated by counting number of occurances in the list passed
def word_prob(words_list,word):
    word_count = words_list.count(word)/len(word_list)
    return word_count

#Function to clean a word
#Cleaning step involves converting all the strings to lowercase and removing punctuations
def clean(s):
    translator = str.maketrans("", "", string.punctuation)
    return s.translate(translator).lower()

#Function to clean tweets
def data_clean(data):
    data2=[]
    for d in data:
        d2=[]
        for word in d:
            #print(word)
            w2 = clean(word)
            #w3 = re.split("\W+", w2)
            #print('w3:',w3)
            d2 .append(w2)
        data2.append(d2)
    return data2    

#os.chdir('C:/Users/singh/Documents/Acads/Elements of AI/Assignment/A2/jaikumar-singhvis-sisath-a2/part2/')

training_file = sys.argv[1]
testing_file = sys.argv[2]

#training_file = str(training_file) + '.txt'
#testing_file = str(testing_file) + '.txt'

#Reading Train file
tweets_train = open(training_file, encoding="latin-1")
train=[]

for line in tweets_train:
    rows = line.split()
    train.append([r for r in rows])

tweets_train.close()

##Reading Test file
tweets_test = open(testing_file, encoding="latin-1")
test=[]

for line in tweets_test:
    rows = line.split()
    test.append([r for r in rows])

tweets_test.close()


location=[]
for t in train:
    if len(t)>0:
        l = t[0]
        location.append(l)

classes = list(set(location))

classes2=[]

#defining location classes
for c in classes:
    if c[len(c)-4:len(c)-2]==',_':
        classes2.append(c)
        
classes2.append('Toronto,_Ontario')

train2 = []
for t in train:
    if len(t)>0:
        if t[0] in classes2:
            train2.append(t)
            
target=[]
for t in train2:
    target.append(t[0])

  
data=[]
for t in train2:
    d=t[1:len(t)]
    data.append(d)

#data=np.array(data)

data_train = data_clean(data)
data_train = np.array(data_train)


#Creating bag of words 
bow=[]
for x in data_train:
    bow+=x 
    
#Selecting unique list of words
bag_of_words2=list(set(bow))


#Calculating probability of each location
p_class={}
for c in classes2:
    p_class[c]=target.count(c)/len(target)
            
prior = pd.DataFrame.from_dict(p_class, orient='index')

#Creating dataframe of likelihhod estimations (p(w|l))
#This is done by calculating the frequency of words that appear for each location.
#The words that do not appear for a particular location are assigned a very small
#frequency (Laplace Smoothing), this value is equal to alpha=0.001 over the number
#of unique words in the bag of words(48625). The frequencies have been converted 
# to log of probabilities of the word for a given location.The dataframe also 
#contains a column 'max_location' which has the value for the location whose 
#frequency is the highest for a particular word. This is used to get the top 5 
#'distinctive' words for a particular location.
bag_of_words_classes={}
target=np.array(target)
for c in classes2:
    cw2=[]
    cw=data_train[np.where(target==c)]
    for l in cw:
        cw2+=l
    bow_c = dict(Counter(cw2))
    bow_c = {k: math.log(v / total) for total in (sum(bow_c.values()),) for k, v in bow_c.items()}
    bag_of_words_classes[c]=bow_c

likelihood_df = pd.DataFrame(bag_of_words_classes)
likelihood_df = likelihood_df.fillna(math.log(0.001/48625))
likelihood_df['max_location'] = likelihood_df.idxmax(axis=1)
likelihood_df['max_likelihood'] = likelihood_df.max(axis=1)

#Calculating top 5 words for each location
#This is done using checking for every word, for which location its frequency
#(probability) is the highest. This is stored in the column 'max_location' along
#with the frequency of the word in the column 'max_likelihood'. The words are then
#grouped by 'max_location' column and top 5 are chosen based on their 'max_likelihood'
#value.
top5_df = likelihood_df.sort_values(['max_location','max_likelihood'], ascending=[True, False]).groupby('max_location').head(5)

top5_dict = {}
for c in classes2:
    top5_dict[c]=list(top5_df[top5_df['max_location']==c].index.values)
    

#Data prep for test dataset
test2 = []
for t in test:
    if len(t)>0:
        if t[0] in classes2:
            test2.append(t)
            
target_test=[]
for t in test2:
    target_test.append(t[0])

  
data_test=[]
for t in test2:
    d=t[1:len(t)]
    data_test.append(d)

data_test=np.array(data_test)

data_test2 = data_clean(data_test)
data_test2 = np.array(data_test2)

#Predicting on test dataset i.e. predicting test location
#prediction is done for a particular word using the formula P(l|w)=P(l).P(w|l) 
#where P(l) is stored in the variable 'p_class' and P(w|l) is obtained from the
#dataframe 'likelihood_df'. Given a set of words in a tweet, P(l|w1 w2 w3 .. wn)
#is calculated for a location using the formula:
#P(l|w1 w2 w3 ... wn) = P(l)P(w1|l).P(w2|l).P(w3|l)....P(wn|l)
#Taking log on both sides
#log(P(l|w1 w2 w3 ... wn)) = log(P(l))+log(P(w1|l))+log(P(w2|l))+ ... + log(P(wn|l))
#We calculate the log of probability for each loaction and take the location with 
#best score 
likelihood_df2 = likelihood_df.iloc[:,0:12]
pred_test=[]
for d in data_test2:
    try:
        pred_df=likelihood_df2.reindex(d)
        pred_df=pred_df.fillna(math.log(0.001/48625))
        pred_df=pred_df.add(prior.values[:,0], axis=1)
        loc = pred_df.sum().idxmax()
        pred_test.append(loc)
    except:
        loc = 'Manhattan,_NY'
        pred_test.append(loc)
    
#checking accuracy on test

acc=0
for i in range(len(pred_test)):
    if pred_test[i]==target_test[i]:
        acc+=1
        
#print('accuracy:',acc*100/len(data_test2))

#print('Top 5 words associated with each location')
for key, value in top5_dict.items():
    print(key,' '.join(value))

#Creating output variable
output_file=[]
for i in range(len(test2)):
    op = [pred_test[i]]+[target_test[i]]+data_test[i]
    op = ' '.join(op)
    output_file.append(op)
    
#creating output file
with open('output-file.txt', 'w', encoding="latin-1") as f:
    for item in output_file:
        f.write(item + '\n')

end = time.time()
#print(end - start)

