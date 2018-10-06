'''
Get counted word sentiment weight of words from Small Phrases
'''

from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from re import sub
import numpy as np
import string
import copy
import pandas as pd

print("Loading data...")
train = pd.read_csv("./datasets/kaggle_movie/train.tsv", sep="\t")
print("Train shape:", train.shape)

np.set_printoptions(precision=10)  # Numpy Precision when Printing

# Get Sentiment Words from a generic Opinion Lexicon
pos_words = []
neg_words = []
for line in open('./opinion_lexicon/positive-words.txt', 'r'):
    pos_words.append(line.rstrip())  # Must strip Newlines

for line in open('./opinion_lexicon/negative-words.txt', 'r'):
    neg_words.append(line.rstrip())  # Must strip Newlines  

wnl = WordNetLemmatizer()
translator = str.maketrans('','', sub('\'', '', string.punctuation))

combinedEmotional = pos_words + sorted(set(neg_words) - set(pos_words))

combinedEmotionalDict = dict()
for i in range(0, len(combinedEmotional)):
    combinedEmotionalDict[combinedEmotional[i]] = [0, 0]  # First zero refers to Count, second zero refers to Sum/Count 

for index, row in train.iterrows():
    temp = []
    for t in word_tokenize(row['Phrase']):
        x = t.translate(translator) 
        if x != '': temp.append(wnl.lemmatize(x.lower())) 
    
    if (len(temp) <= 4):
        for i in range(0, len(temp)):
            if (temp[i] in combinedEmotionalDict):            
                combinedEmotionalDict[temp[i]][0] += 1
                combinedEmotionalDict[temp[i]][1] += row['Sentiment']
    
todelete = []
for key, value in combinedEmotionalDict.items():
    if (value[0] == 0):  # If Count is 0
        todelete.append(key) 
    else:
        combinedEmotionalDict[key][1] = value[1] / value[0]

for k in todelete: del combinedEmotionalDict[k]

output_file = pd.DataFrame.from_dict(combinedEmotionalDict, orient='index', columns=['Count', 'Sum/Count'])
output_file.to_csv('specialWeights.csv', index=True, sep="\t")


quit()