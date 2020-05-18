import pandas as pd
import pickle
from collections import Counter

import itertools
flatten = itertools.chain.from_iterable

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

def model(publications_df, keys_list_year, list_of_years, n_years, n_samples):
    list_of_years.sort()
    
    publications_df = publications_df.rename(columns={keys_list_year[0]:'clean_body_list_of_terms', keys_list_year[1]: 'year'})
    publications_df = publications_df.head(n_samples)
    
    #Separate years to t_1 and t
    t_1 = list_of_years[1:] #t+1 year
    t = list_of_years[:-1]  #t year

    #Extract unique terms (after stopword removal and lemmatization)
    print(publications_df.columns)

    list_of_terms = publications_df["clean_body_list_of_terms"].tolist()
    unique_terms = list(set(list(flatten(list_of_terms))))

    #1. FrequencyAnalysis
    #a. Count term cumulatively up until certain year
    terms_frequency_df = pd.DataFrame(unique_terms,columns=['Term'])
    str_year_ctr_list = []
    for year in list_of_years:
        str_year_ctr = "Count:Year:" + str(year)
        print(str_year_ctr)
        str_year_ctr_list.append(str_year_ctr)
        sub_publications_df = publications_df[(publications_df['year'] <= year)]
        list_of_terms_sub = sub_publications_df["clean_body_list_of_terms"].tolist()
        list_of_terms_sub = list(flatten(list_of_terms_sub))
        term_counter = Counter(list_of_terms_sub)
        term_counter_df = pd.DataFrame.from_dict(term_counter, orient='index').reset_index()
        term_counter_df = term_counter_df.rename(columns={'index':'Term', 0:str_year_ctr})
        terms_frequency_df = pd.merge(left=terms_frequency_df, right=term_counter_df, left_on='Term', right_on='Term')
    
    #b. Compute frequency divergence
    str_year_div_list = []
    for t0, t1 in zip(str_year_ctr_list[:-1], str_year_ctr_list[1:]):
        year_t0 = t0.split(':')[-1] #Year t0
        year_t1 = t1.split(':')[-1] #Year t1
        str_year_ctr = "Count:Year:" + str(year_t1[-1])
        str_year_div = "Divergence:Year:" + str(year_t0) + ':' + str(year_t1)
        print(str_year_div)
        str_year_div_list.append(str_year_div)
        terms_frequency_df[str_year_div] = (terms_frequency_df[t0] - terms_frequency_df[t1]).abs()  #|c_{t+1}(w) - c_{t}|
        terms_frequency_df[str_year_div] = terms_frequency_df[str_year_div] / terms_frequency_df[t1] #divide by c_{t}

    #c. Compute average cumulative frequency divergence (divide by n_years)
    terms_frequency_df["AverageSumDivergence"] = terms_frequency_df[str_year_div_list].sum(axis=1)  
    terms_frequency_df["AverageSumDivergence"] = terms_frequency_df["AverageSumDivergence"].divide(n_years)

    return terms_frequency_df


#%% Cord19
print("Cord19")
# publications_path = '/home/syafiq/Developments/temporal_summarization/assets/publications.p'
# publications_df = pickle.load(open(publications_path, "rb")) 
# keys = ['clean_body_list_of_terms', 'year']
# publications_df["clean_body_list_of_terms"] = publications_df['body_list_of_terms'].apply(lambda terms:[lemmatizer.lemmatize(token) for token in terms if not token in stop_words])

# list_of_years = [1990,2000,2010,2020]
# total_years = 2020 - 1950
# total_publications = 50000

# termFrequencyModel_df = model(publications_df, keys, list_of_years, total_years, total_publications)
# pickle.dump(termFrequencyModel_df, open( "cord19_termFrequencyModel.p", "wb" ))

#%% ACL
print("ACL")
publications_path = '/home/syafiq/Developments/temporal_summarization/assets/acl_publications.p'
publications_df = pickle.load(open(publications_path, "rb")) 
keys = ['Lemmatized_Body_Tokens_List', 'Year']
list_of_years = [year for year in range(2010,2016)]
total_years = 2015-1979
total_publications = 19905

termFrequencyModel_df = model(publications_df, keys, list_of_years, total_years, total_publications)
pickle.dump(termFrequencyModel_df, open( "./assets/acl_termFrequencyModel.p", "wb" ))