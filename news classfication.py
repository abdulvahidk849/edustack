import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, KFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

import time
import re
import pickle
from string import punctuation
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from wordcloud import WordCloud ,STOPWORDS

df = pd.read_json('News_Category_Dataset_v2.json', lines=True)
df.head()

df.isna().sum()

df['text'] = df['headline'] + df['short_description'] + df['authors']
df['label'] = df['category']
del df['headline']
del df['short_description']
del df['date']
del df['authors']
del df['link']
del df['category']

df['News_length'] = df['text'].str.len()
print(df['News_length'])

def process_text(text):
    text = text.lower().replace('\n',' ').replace('\r','').strip()
    text = re.sub(' +', ' ', text)
    text = re.sub(r'[^\w\s]','',text)
    
    
    stop_words = set(stopwords.words('english')) 
    word_tokens = word_tokenize(text) 
    filtered_sentence = [] 
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
    
    text = " ".join(filtered_sentence)
    return text

df['text'] = df['text'].apply(process_text)

X=df.text
y=df.label

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# Creating Models
models = [('Logistic Regression', LogisticRegression(max_iter=500)),('Random Forest', RandomForestClassifier()),
          ('Linear SVC', LinearSVC()), ('Multinomial NaiveBayes', MultinomialNB()), ('SGD Classifier', SGDClassifier())]

names = []
results = []
model = []
for name, clf in models:
    pipe = Pipeline([('vect', CountVectorizer(max_features=30000, ngram_range=(1, 2))),
                    ('tfidf', TfidfTransformer()),
                    (name, clf),
                    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    
    names.append(name)
    results.append(accuracy)
    model.append(pipe)
    
    msg = "%s: %f" % (name, accuracy)
    print(msg)

# Logistic Regression
filename = 'model_lr.sav'
pickle.dump(model[0], open(filename, 'wb'))


lr_model = pickle.load(open('model_lr.sav', 'rb'))
text1 = 'Royal Challengers Bangalore beat Mumbai Indians in Super Over'
text2 = "Microsoft Surface Laptop Go With 12.5-Inch Display, Revamped Surface Pro X to Launch on October 1"
print(lr_model.predict([text1, text2]))
