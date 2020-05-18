import numpy as np
import pandas as pd
import pickle

from sklearn.manifold import TSNE
from gensim.models import Word2Vec
 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import math

matplotlib.rcParams.update({'font.size': 45})
matplotlib.rcParams['figure.figsize'] = (30,30)

def clusterClosestWords_tsne(word, models, years):
    Colors = []
    Labels = []
    Xs = []
    Ys = []
    
    for model, year in zip(models,years):
        vector_dim = model.vector_size
        arr = np.empty((0,vector_dim), dtype='f')
        theword = word + "\n(" + str(year) + ")"
        word_labels = [theword]



        # get close words
        close_words = model.wv.similar_by_word(word, topn=3)
        print(year, close_words)
        
        # add the vector for each of the closest words to the array
        arr = np.append(arr, np.array([model.wv[word]]), axis=0)
        for wrd_score in close_words:
            wrd_vector = model.wv[wrd_score[0]]
            word_labels.append(wrd_score[0])
            arr = np.append(arr, np.array([wrd_vector]), axis=0)
            
        # find tsne coords for 2 dimensions
        tsne = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        Y = tsne.fit_transform(arr)
    
        x_coords = Y[:, 0]
        y_coords = Y[:, 1]
        
        colors = ['b' for i in range(len(x_coords))]
        colors[0] = 'r'
        
        #Append to list
        Labels.append(word_labels)
        Xs.append(x_coords)
        Ys.append(y_coords)
        Colors.append(colors)
    
    return Xs,Ys,Labels,Colors


def singleplotScatterCluster(data):
    year_to_use = [1990,1995,2000,2005,2010,2015]
    
    Xs = data[0]
    Ys = data[1]
    XYLabels = data[2]
    XYColors = data[3]
    theword = XYLabels[0][0][:-7]
    
    title = 'The semantic divergence for word "' + theword + '"'
    plt.title(title)
    # plt.figure(figsize=(30,30))
    for xs, ys, labels, clrs in zip(Xs, Ys, XYLabels, XYColors): 
        plt.scatter(xs, ys, color=clrs, s=300)
        for label, x, y in zip(labels, xs, ys):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points') 
    plt.show()


list_of_years = [1990,1995,2000,2005,2010,2015]
w2v_paths = [
    '/home/syafiq/Developments/w2v_acl/1990.w2v',
    '/home/syafiq/Developments/w2v_acl/1995.w2v',
    '/home/syafiq/Developments/w2v_acl/2000.w2v',
    '/home/syafiq/Developments/w2v_acl/2005.w2v',
    '/home/syafiq/Developments/w2v_acl/2010.w2v',
    '/home/syafiq/Developments/w2v_acl/2015.w2v',
]

#Sort in ascending order
list_of_years.sort()
w2v_paths.sort()

#1. Load the word2vec model
w2v_models = [Word2Vec.load(path) for path in w2v_paths]

data_to_plot = clusterClosestWords_tsne('cps', w2v_models, list_of_years)
singleplotScatterCluster(data_to_plot)