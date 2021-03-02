#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('treebank')
nltk.download('stopwords')

from nltk.draw.tree import draw_trees


# In[2]:


# load python using pandas
filename = 'datasets/data_in.xlsx'
excel_data = pd.read_excel(filename)


# In[3]:


data_to_string = pd.DataFrame.to_string(excel_data) 
print(data_to_string)


# In[4]:


def sent_tokenize_text(excel_data):
    for columns in excel_data.columns:
        dataframe = excel_data.apply(lambda row: nltk.sent_tokenize(row[columns]), axis=1)
    return dataframe


# In[5]:


sent_tokenized_data = sent_tokenize_text(excel_data)
sent_out_data = pd.DataFrame(sent_tokenized_data, columns = ['Sent_Tokenized_Comment'])
sent_out_data.head()


# In[6]:


def word_tokenize_text(excel_data):
    for columns in excel_data.columns:
        dataframe = excel_data.apply(lambda row: nltk.word_tokenize(row[columns]), axis=1)
        
    return dataframe


# In[7]:


excel_data.head()
word_tokenized_data = word_tokenize_text(excel_data)
word_out_data = pd.DataFrame(word_tokenized_data, columns = ['Word_Tokenized_Comment'])
word_out_data.head()


# In[8]:


out_data = excel_data.join(sent_out_data).join(word_out_data)
out_data.head()
out_data.to_excel('datasets/output/data_out.xlsx')


# In[9]:


def pos_tokenize_word(excel_data):
    for columns in excel_data.columns:
        dataframe = excel_data.apply(lambda row: nltk.pos_tag(row[columns]), axis=1)
    return dataframe


# In[10]:


import os
from IPython.display import Image


# In[11]:


txtfile = 'datasets/NLPdataEx3&4-data_in.txt'
file1 = open(txtfile, 'r')
Lines = file1.readlines()


for line in Lines:
    # Strips the newline character
    curline = line.strip()
    
    #word tokenize the line
    wordtokenized = nltk.word_tokenize(curline)
    
    #pos tag the line
    postagged = nltk.pos_tag(wordtokenized)
    
    #build grammer and RegEx parser
    grammar = "NP: {<DT>?<JJ>*<NN>}"
    cp = nltk.RegexpParser(grammar)
    
    #build the POS tagged result
    result = cp.parse(postagged)
    
    print(result)


# In[12]:


import io  
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize  
  
# word_tokenize accepts 
# a string as an input, not a file.  
stop_words = set(stopwords.words('english'))  
file1 = open("datasets/NLPdataEx5data_senti_analyze.txt")  


# Use this to read file content as a stream:  
lines = file1.readlines() 


# In[13]:


out_array = []
for line in Lines:
    words = line.split()
    s=''
    for r in words:  
        if not r in stop_words:  
            s+= r
            s+= ' '

    out_array.append(s)

print(out_array)

list_of_tuples2 = list(zip(Lines, out_array))  
    
# Assign data to tuples.  
list_of_tuples2
# Converting lists of tuples into  
# pandas Dataframe.  
df = pd.DataFrame(list_of_tuples2, 
                  columns = ['Original', 'Stopwords removed'])
df.head()
df.to_csv('datasets/output/stopword_removed.csv')


# In[14]:


from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

porter=PorterStemmer()

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)


# In[15]:


stemmed_array = []
for line in Lines:
    x=stemSentence(line)
    stemmed_array.append(x)

print(stemmed_array)

list_of_tuples = list(zip(Lines, stemmed_array))  
    
# Assign data to tuples.  
list_of_tuples   
# Converting lists of tuples into  
# pandas Dataframe.  
df = pd.DataFrame(list_of_tuples, 
                  columns = ['Original', 'Stemmed'])
df.head()
df.to_csv('datasets/output/stemmed.csv')


# In[16]:


import io  
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize  
  
# word_tokenize accepts 
# a string as an input, not a file.  
stop_words = set(stopwords.words('english'))  
file1 = open("datasets/NLPdataEx3&4-data_in.txt")  


# Use this to read file content as a stream:  
lines = file1.readlines() 


# In[25]:


from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()


def lemmatizeSentence(sentence):
    token_words=word_tokenize(sentence)
    #pos tag the line
    lemmatize_sentence=[]
    for word in token_words:
        lemmatize_sentence.append(wordnet_lemmatizer.lemmatize(word, 'v'))
        lemmatize_sentence.append(" ")
    return "".join(lemmatize_sentence)


# In[26]:


lemmatized_array = []
for line in Lines:
    x=lemmatizeSentence(line)
        
    lemmatized_array.append(x)

print(lemmatized_array)

list_of_tuples_1 = list(zip(Lines, lemmatized_array))  
    
# Assign data to tuples.  
list_of_tuples   
# Converting lists of tuples into  
# pandas Dataframe.  
df = pd.DataFrame(list_of_tuples_1, 
                  columns = ['Original', 'Lemmatized'])
df.head()
df.to_csv('datasets/output/lemmatized.csv')


# In[27]:


senti_dict = {}

for each_line in open('datasets/my_dict.txt'):
    word, score = each_line.split('\t')
    senti_dict[word] = int(score)
    


# In[28]:


for line in open('datasets/NLPdataEx5data_senti_analyze.txt'):
    x=lemmatizeSentence(line)
    words = x.split()
    print(words)
    print(sum(senti_dict.get(word,0) for word in words))


# In[ ]:




