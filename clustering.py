#Clustering
from sklearn.cluster import KMeans
import networkx as nx
from chinese_whispers import chinese_whispers, aggregate_clusters
from sklearn.cluster import AffinityPropagation


#Pandas
import pandas as pd
import pickle
from collections import Counter
import numpy as np

#Word2Vec
from gensim.models import Word2Vec

def model(terms, list_of_years, list_w2v_path):
    def kMeanClustering(series):
        vectors = series.tolist()
        #Clustering
        kmeans  = KMeans(n_clusters=4, n_jobs=-1)
        kmeans.fit(vectors)
        
        #Cluster
        y_kmeans = kmeans.predict(vectors)
        return y_kmeans
    
    def affinityClustering(series):
        vectors = series.tolist()
        #Clustering
        affinity  = AffinityPropagation()
        affinity.fit(vectors)
        
        #Cluster
        y_affinity = affinity.predict(vectors)
        return y_affinity

    #1. Get all the vectors
    #a. Load the word2vec model
    w2v_models = [(t,Word2Vec.load(m)) for t,m in zip(list_of_years,list_w2v_path)]
    w2v_models = dict(w2v_models)

    #b. Load the word2vec model to each term
    terms_df = pd.DataFrame(data=terms, columns=['Term'])
    for year in list_of_years:
        str_kmean_cluster_year = "KMean:Cluster:Year:" + str(year)
        str_affinity_cluster_year = "Affinity:Cluster:Year:" + str(year)
        
        terms_df[str_kmean_cluster_year] = terms_df['Term'].apply(lambda term:w2v_models[year].wv[term])
        terms_df[str_kmean_cluster_year] = kMeanClustering(terms_df[str_kmean_cluster_year])

        terms_df[str_affinity_cluster_year] = terms_df['Term'].apply(lambda term:w2v_models[year].wv[term])
        terms_df[str_affinity_cluster_year] = affinityClustering(terms_df[str_affinity_cluster_year])

    #c. Return the terms with cluster
    return terms_df

#%% Cord 19
# print("Cord19")
# termFrequency_df = pickle.load(open('assets/termFrequencyModel.p', 'rb'))
# termSemantic_df = pickle.load(open('assets/cord19_termSemanticModel.p', 'rb'))

# all_df = pd.merge(left=termFrequency_df, right=termSemantic_df, on='Term', how='inner')
# all_df = all_df[(all_df['Count:Year:2020'] > 100)]  #We only care term that has more than 100 count
# terms = all_df['Term'].tolist()
# w2v_paths = [
#     '/home/syafiq/Developments/w2v/1990.w2v',
#     '/home/syafiq/Developments/w2v/2000.w2v',
#     '/home/syafiq/Developments/w2v/2010.w2v',
#     '/home/syafiq/Developments/w2v/2020.w2v'
# ]
# list_of_years = [1990,2000,2010,2020]
# clusters = model(terms, list_of_years,w2v_paths)
# pickle.dump(clusters, open( "cord19_clusters.p", "wb" ))

#%% ACL Cluster
print("ACL")
termFrequency_df = pickle.load(open('assets/acl_termFrequencyModel.p', 'rb'))
termSemantic_df = pickle.load(open('assets/acl_termSemanticModel.p', 'rb'))

all_df = pd.merge(left=termFrequency_df, right=termSemantic_df, on='Term', how='inner')
all_df = all_df[(all_df['Count:Year:2015'] > 100)]  #We only care term that has more than 100 count
terms = all_df['Term'].tolist()
w2v_paths = [
    '/home/syafiq/Developments/w2v_acl/1990.w2v',
    '/home/syafiq/Developments/w2v_acl/1995.w2v',
    '/home/syafiq/Developments/w2v_acl/2000.w2v',
    '/home/syafiq/Developments/w2v_acl/2005.w2v',
    '/home/syafiq/Developments/w2v_acl/2010.w2v',
    '/home/syafiq/Developments/w2v_acl/2015.w2v',
]
list_of_years = [1990,1995,2000,2005,2010,2015]
clusters = model(terms, list_of_years,w2v_paths)
pickle.dump(clusters, open( "acl_clusters.p", "wb" ))