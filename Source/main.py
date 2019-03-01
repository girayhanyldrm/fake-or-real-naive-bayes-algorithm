from heapq import nlargest
import  pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import math

''' chose type unigram or bigram '''
def chose_type(type):
    all_data_value = len(fake_list) + len(real_list)
    p_fake = len(fake_list) / all_data_value
    p_real = len(real_list) / all_data_value
    if (type == 'unigram'):
        V = count_of_unique_words(fake_dic, real_dic)
        return V,p_fake,p_real
    else:
        V = count_of_unique_words(fake_dic2, real_dic2)
        return V,p_fake,p_real
''' count of unique words'''
def count_of_unique_words(dict1,dict2):
    common_value=dict1.keys() & dict2.keys()
    return len(dict1.keys())+len(dict2.keys())-len(common_value)
def bigram_format(text):
    cv = CountVectorizer(ngram_range=(2, 2))
    cv.fit_transform([text])
    return cv.get_feature_names()
''' creating word dictionary with frequency  '''
def dictionary(liste,n):
    dictionary={}
    vectorizer = CountVectorizer(ngram_range=(n, n))
    X = vectorizer.fit_transform(liste).toarray()
    features = vectorizer.get_feature_names()
    count=sum(X)
    for i in range(0,count.__len__()):
        dictionary[features[i]]=count[i]
    return dictionary

''' calculating probability '''
def probability_fake(fake_dic,word,v):
    x=(fake_dic.get(word,0)+1)
    return x/(len(fake_dic)+v)

def probability_real(real_dic,word,v):
    x=(real_dic.get(word,0)+1)
    return x/(len(real_dic)+v)

''' naive bayes function '''
def naive_bayes(text,value,V,p_fake,p_real,fake_dict,real_dict):
    for word in text:

        p_fake*=probability_fake(fake_dict,word,V)
        p_real*=probability_real(real_dict,word,V)

    if(p_fake>=p_real):
        return ["fake",value]
    else:
        return ["real",value]

def read_data(data_file):
    read=open(data_file)
    list=[]
    list2=[]
    for line in read:
        list.append(line)
        x=line.replace("\n","").split(" ")
        list2.append(x)
    return list,list2


def presence(fake_dic,real_dic,p_fake,p_real):

    fake_presence={}
    real_presence={}
    for key,value in fake_dic.items():
        payda=(fake_dic.get(key)/len(fake_list))+(real_dic.get(key,0.00001)/len(real_list))
        probability=(fake_dic.get(key)/len(fake_list))*p_fake/payda
        fake_presence[key]=probability
    for key,value in real_dic.items():

        payda=(real_dic.get(key)/len(real_list))+(fake_dic.get(key,0.00001)/len(fake_list))
        probability=((real_dic.get(key)/len(real_list))*p_real)/payda
        real_presence[key]=probability
    return fake_presence,real_presence

def word_without_headlines(word):
    count=0
    total_headlines= fk2+rl2
    for i in range(0,len(total_headlines)):
        if word not in total_headlines[i]:
            count+=1
    return count/len(total_headlines)

def absence(fake_dic,real_dic,p_fake,p_real):

    fake_absence={}
    real_absence={}
    for word in total_words:
        payda=word_without_headlines(word)
        probability=(1-(fake_dic.get(word,0.1)/len(fake_list)))*p_fake/payda
        fake_absence[word]=probability

        payda = word_without_headlines(word)
        probability = (1 - (real_dic.get(word, 0.1) / len(real_list))) * p_real / payda
        real_absence[word] = probability

    return fake_absence,real_absence

'''###      READ TRAIN DATA AND CREATING DICTIONARY      ###'''

fake_list,fk2=read_data('clean_fake-Train.txt')
real_list,rl2=read_data('clean_real-Train.txt')
fake_dic=dictionary(fake_list,1)
real_dic=dictionary(real_list,1)
fake_dic2=dictionary(fake_list,2)
real_dic2=dictionary(real_list,2)
test= pd.read_csv('test.txt', sep=",", header=None)
test.columns=["Id","Category"]
'''  chose type '''
type='unigram'
V,p_fake,p_real=chose_type(type)
total_words=list(set(list(real_dic.keys())+list(fake_dic.keys())))

'''###      MAIN        ###'''
acc=0
for index,row in test.iterrows():
    if(index!=0):
        if(type=='unigram'):
            text=row[0].split(" ")
            liste=naive_bayes(text,row[1],V,p_fake,p_real,fake_dic,real_dic)
            if(liste[0]==liste[1]):
                acc+=1
        else:
            liste=naive_bayes(bigram_format(row[0]),row[1],V,p_fake,p_real,fake_dic2,real_dic2)
            if(liste[0]==liste[1]):
                acc+=1

print(acc/(len(test)-1)*100)

'''###  CALCULATE PRESENCE AND ABSENCE   ###'''
#fake_pre,real_pre=presence(fake_dic,real_dic,p_fake,p_real)
#print(nlargest(10, fake_pre, key=fake_pre.get))
#print(nlargest(10, real_pre, key=real_pre.get))
#fake_absence,real_absence=absence(fake_dic,real_dic,p_fake,p_real)
#print(nlargest(10, fake_absence, key=fake_absence.get))
#print(nlargest(10, real_absence, key=real_absence.get))
