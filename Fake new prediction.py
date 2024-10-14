'''
Id: unique id for a news article
Title: the title of a news article
Author: author of the news article
Text: the text of the article; could be incomplete
Label: a label that marks the article as potentially unreliable
1: fake news
0: real  news
'''
#pip install PorterStemmer
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords #stopwords means words which doesnt vales to our text(like wt where how) ,we will remove all such words
from nltk.stem.porter import PorterStemmer #stem words(give root words to a particular word)
from sklearn.feature_extraction.text import TfidfVectorizer #converting text feature_vectors(numbers)
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk #natural language toolkit
nltk.download('stopwords')

#printing the stopwords in english
print(stopwords.words('english'))

#data Pre-processing
#loading the dataset into pandas
news_dataset = pd.read_csv(r"D:\fakes news projects\train.csv")

news_dataset.shape

#printing first 5 rows
news_dataset.head()

#counting the number of missing values
news_dataset.isnull().sum()

#replacing the null values with empty string
news_dataset = news_dataset.fillna('')

'''
we combine(merging) title and author and use this data for prediction which give very good peformance score
 , we dont use text bcs they are too large
'''
news_dataset['content'] = news_dataset['author']+" "+ news_dataset['title']

print(news_dataset['content'])

#we use content column and label for prediction
#seperating the data & label

x = news_dataset.drop('label' , axis=1 )

y = news_dataset['label']

print(x) #data
print(y) #label

#stemming
''' stemming is the process of reducing a word to a root word
    example:- actor,actress,acting , so root word is act'''
    
port_stem = PorterStemmer()

#function we r using for stemming our txt data

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content) #searching ,^ means exclusion (we need oly words, neglecting any num present in text)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split() #split the txt and converting into list
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)

print(news_dataset['content'])

#seperating the data and label
x = news_dataset['content'].values
y = news_dataset['label'].values

print(x)

print(y)

#converting textual data to numerical data

vectorizer = TfidfVectorizer() 

'''tf = trem frequency and idf = inverse document frequency 
(basically it counts number of times a particular word is repeating  the document or txt paragraph)
so the repetiton tells the model that it is a imp word as it assigns a numerical value to that word'''

vectorizer.fit(x)

x = vectorizer.transform(x) #transfrom will convert the values to respective features

print(x)

#spllintg the data into tranning and test data

x_train, x_test, y_train, y_test = train_test_split(x , y, test_size=0.2, stratify=y, random_state=2)

#training the model: logistic regression

model = LogisticRegression()
''' shape of graph is like S so we call it sigmoid function 
    and it is given by Y=1/1+e^-z , here we use threshold values as 0.5 and if the prediction is 
    greater then 0.5 it gives value as 1 and vise-versa'''
    
model.fit(x_train,y_train)

#evaulation

'''accuracy score'''

#accuracy score on the traning data

x_train_prediction = model.predict(x_train)
traning_data_accuracy = accuracy_score(x_train_prediction, y_train)

print('accuracy score of traning data : ', traning_data_accuracy)

#accuracy score on the test data

x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)

#making a predictive system
x_new = x_test[1]

prediction = model.predict(x_new)
print(prediction)

if(prediction[0]==0):
    print('The news is Real')
else:
    print('The news is Fake')

print(y_test[1])



               
               
               
               
               


























