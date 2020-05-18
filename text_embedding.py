#%% Building vocabulary (text preprocessing tokenize, lemmatize)
from gensim.test.utils import common_texts, get_tmpfile
import multiprocessing
from gensim.models import Word2Vec
from time import time  # To time our operations
import pandas as pd
import pickle

def learn_temporal_embedding(title, publications_df, key_list_tokens_year, list_of_years):
    path_to_w2v = "../" + title + "/"
    #1. Initialized Gensim
    w2v_model = Word2Vec(min_count=10,
                window=5,
                size=300,
                sample=6e-5, 
                alpha=0.03, 
                min_alpha=0.0007, 
                negative=10,
                workers=multiprocessing.cpu_count())

    #2. Build entire vocabulary
    t = time()
    w2v_model.build_vocab(publications_df[key_list_tokens_year[0]].tolist())
    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
    if 'finaly' in w2v_model.wv.vocab:
        print("Yes")
        return 0

    modelfile = ""
    prev_years = []
    path_to_w2v_models = []
    for year in list_of_years:
        print("Training up to time unit:" + str(year))
        if len(prev_years) == 0:
            sub_publications_df = publications_df[(publications_df[key_list_tokens_year[1]] <= year)]
            modelfile = str(year) + ".w2v"
            prev_years.append(year)
        elif len(prev_years) > 0:
            sub_publications_df = publications_df[(publications_df[key_list_tokens_year[1]] <= year) &(publications_df[key_list_tokens_year[1]] > prev_years[-1])]
            modelfile = str(year) + ".w2v"
            prev_years.append(year)
        w2v_model.train(sub_publications_df[key_list_tokens_year[0]].tolist(), total_examples=len(sub_publications_df.index), epochs=30, report_delay=1)
        path_model_file = path_to_w2v + modelfile
        w2v_model.save(path_model_file)
        path_to_w2v_models.append(path_model_file)
    return path_to_w2v_models

#%% Acl publications
publications_df = pickle.load(open('assets/acl_publications.p', 'rb'))
key_list_tokens_years = ['Lemmatized_Body_Tokens_List','Year']
list_of_years = [year for year in range(1990,2016)] #We are going to train up to every unit time based on yearly interval
models_paths = learn_temporal_embedding('w2v_acl',publications_df,key_list_tokens_years,list_of_years)