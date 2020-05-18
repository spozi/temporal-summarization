import xmltodict
import json
import pandas as pd
import os
import xml.etree.ElementTree as ET
import datetime
import itertools
import pickle
flatten = itertools.chain.from_iterable

#Text preprocessing
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
stop_words = list(set(stopwords.words('english')))
stop_words.extend(["et", "al", "de", "fig", "en", "use"])

pickle.dump(stop_words, open("stopwords.p", "wb"))

lemmatizer = WordNetLemmatizer() 
tokenizer = RegexpTokenizer(r'\w+')

acl_dir = "/media/syafiq/Seagate/datasets/acl/acl-arc.comp.nus.edu.sg/archives/acl-arc-160301-parscit/"

xml_files = [] #We store all the publications based json files into here.
for dirname, _, filenames in os.walk(acl_dir):
    for filename in filenames:
        filepath = os.path.join(dirname, filename)
        extension = os.path.splitext(filepath)[1]
        if os.path.splitext(filepath)[1] == '.xml':
            xml_files.append(filepath)

publications = []
for filepath in xml_files:
    #1. Extract year
    year = filepath.split('/')  #a/E09/E09-1009-parscit.130908.xml
    year = year[-2]             # return E09
    year = year[1:]             # return '09
    year = datetime.datetime.strptime(year,'%y').strftime('%Y') # return 2009

    try:
        #2. Parse data
        tree = ET.parse(filepath)
        root = tree.getroot()
        title = root.findall('.//algorithm[@name="SectLabel"]/variant/title')
        title = title[0].text
        body_texts = root.findall('.//algorithm[@name="SectLabel"]/variant/bodyText')
        list_body_texts = [itr.text for itr in body_texts]

        #3. Store into panda data
        combined_body_texts = list(flatten(list_body_texts))
        combined_body_texts = ''.join([str(elem) for elem in combined_body_texts])
        publications.append((int(year), title, combined_body_texts))

    except:
        print("Problem occur at file: ", filepath)


acl_publications_df = pd.DataFrame(publications, columns=['Year', 'Title', 'Body_Text'])

#Title
acl_publications_df['Title'] = acl_publications_df['Title'].apply(lambda term: term.replace("-\n", ""))
acl_publications_df['Title'] = acl_publications_df['Title'].apply(lambda term: term.replace("\n", " "))

#Body_Text
acl_publications_df['Body_Text'] = acl_publications_df['Body_Text'].apply(lambda term: term.replace("-\n", ""))
acl_publications_df['Body_Text'] = acl_publications_df['Body_Text'].apply(lambda term: term.replace("\n", " "))

#Tokenize and lemmatize
def lemmatize_sentences(sentences):
    sentences = sentences.lower()
    tokens = tokenizer.tokenize(sentences)
    tokens = [token for token in tokens if not token in stop_words]
    lemmatized_tokens = []
    for token in tokens:
        lemmatized_tokens.append(lemmatizer.lemmatize(token))
    return lemmatized_tokens

#Tokenize
acl_publications_df['Lemmatized_Body_Tokens_List'] = acl_publications_df['Body_Text'].apply(lambda text: lemmatize_sentences(text))
pickle.dump(acl_publications_df, open( "acl_publications.p", "wb" )) 
print("Done")   