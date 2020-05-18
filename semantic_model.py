#%% Semantinc model
import pandas as pd
import pickle
from collections import Counter
import numpy as np

import itertools
flatten = itertools.chain.from_iterable

#Word2Vec
from gensim.models import Word2Vec
#Scipy
from scipy import spatial

#Text preprocessing
import nltk
# nltk.download('all')
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
stop_words = list(set(stopwords.words('english')))
stop_words.extend(["et", "al", "de", "fig", "en", "use"])

pickle.dump(stop_words, open("stopwords.p", "wb"))

lemmatizer = WordNetLemmatizer() 
tokenizer = RegexpTokenizer(r'\w+')

from pandarallel import pandarallel
pandarallel.initialize()

#top highest m terms
#n_et = number of explanatory terms
def model(terms, list_of_years, list_w2v_path, total_years, topm, n_et):
    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def semanticDivergence(a, b):
        return spatial.distance.cosine(a, b)
    
    #Sort in ascending order
    list_of_years.sort()
    list_w2v_path.sort()

    #1. Load the word2vec model
    w2v_models = [(t,Word2Vec.load(model)) for t,model in zip(list_of_years,list_w2v_path)]
    w2v_models = dict(w2v_models)

    #2. Measure the divergence between w(t0) and w(t1)
    terms_semantic_df = pd.DataFrame(data=terms, columns=['Term'])  #Load the file
    str_year_div_list = []
    str_year_dir_list = []
    for t0, t1 in zip(list_of_years[:-1], list_of_years[1:]):
        #a. Divergence
        str_year_div = "SemanticDiv:Year:" + str(t0) + ":" + str(t1)
        str_year_div_list.append(str_year_div)
        terms_semantic_df[str_year_div] = terms_semantic_df['Term'].apply(lambda x: semanticDivergence(w2v_models[t0].wv[x],w2v_models[t1].wv[x]))
        
        #b. Direction
        str_year_dir = "SemanticDir:Year:" + str(t0) + ":" + str(t1)
        str_year_dir_list.append(str_year_dir)
        terms_semantic_df[str_year_dir] = terms_semantic_df['Term'].apply(lambda x: angle_between(w2v_models[t0].wv[x],w2v_models[t1].wv[x]))

    #3. Compute the sum divergence average
    terms_semantic_df['SumDivAverage'] = terms_semantic_df[str_year_div_list].sum(axis=1)
    terms_semantic_df['SumDivAverage'] = terms_semantic_df['SumDivAverage'].divide(total_years)

    #3. Compute the sum direction average
    terms_semantic_df['SumDirAverage'] = terms_semantic_df[str_year_dir_list].sum(axis=1)
    terms_semantic_df['SumDirAverage'] = terms_semantic_df['SumDirAverage'].divide(total_years)

    #4. Term trending
    # terms_semantic_df['DivTrend'] = terms_semantic_df[str_year_dir_list].apply(lambda x: )


    #4. Get the top m most diverge
    terms_semantic_df = terms_semantic_df.sort_values(by='SumDivAverage', ascending=False)
    m_terms_semantic_df = terms_semantic_df.head(topm)
    m_terms_semantic_df = m_terms_semantic_df.copy()

    #4. explanatory terms (watchout for words that are literally similar)
    for year in list_of_years:
        str_year_et = "ExplanatoryTerm:" + str(year)
        m_terms_semantic_df[str_year_et] = m_terms_semantic_df['Term'].apply(lambda term:w2v_models[year].wv.most_similar(term, topn=n_et))

    #4. Return the dataframe
    return m_terms_semantic_df

#%% Cord19
# print("Cord19")
# total_years = 2020 - 1950
# n = 10000   #top n terms with highest averagesumdivergence
# m = 100     #top m terms with highest semantic divergence
# n_et = 10    #explanatory terms

# termsFrequency_df = pickle.load(open('assets/cord19_termFrequencyModel.p', 'rb'))
# termsFrequency_df = termsFrequency_df.sort_values(by='AverageSumDivergence', ascending=False)
# nTermsFrequency_df = termsFrequency_df.head(n)
# terms = nTermsFrequency_df['Term'].tolist()
# list_of_years = [1990,2000,2010,2020]
# w2v_paths = [
#     '/home/syafiq/Developments/w2v/1990.w2v',
#     '/home/syafiq/Developments/w2v/2000.w2v',
#     '/home/syafiq/Developments/w2v/2010.w2v',
#     '/home/syafiq/Developments/w2v/2020.w2v'
# ]

# termSemantic_df = model(terms, list_of_years, w2v_paths, total_years, m, n_et)
# pickle.dump(termSemantic_df, open( "cord19_termSemanticModel.p", "wb" ))

#%% ACL
print("ACL")
total_years = 2015-1979
n = 10000   #top n terms with highest averagesumdivergence
m = 100     #top m terms with highest semantic divergence
min_count = 100 #term must have higher or at least min_count to be compared against
n_et = 10    #explanatory terms

termsFrequency_df = pickle.load(open('./assets/acl_termFrequencyModel.p', 'rb'))
termsFrequency_df = termsFrequency_df[(termsFrequency_df["Count:Year:2015"] >= min_count)]
termsFrequency_df = termsFrequency_df.sort_values(by='AverageSumDivergence', ascending=False)
nTermsFrequency_df = termsFrequency_df.head(n)
terms = nTermsFrequency_df['Term'].tolist()
list_of_years = [year for year in range(1990,2016)]
path = '/home/syafiq/Developments/w2v_acl/'
w2v_paths = [path + str(year) + ".w2v" for year in list_of_years]
# print(w2v_paths[0])
termSemantic_df = model(terms, list_of_years, w2v_paths, total_years, m, n_et)
pickle.dump(termSemantic_df, open( "./assets/acl_termSemanticModel.p", "wb"))