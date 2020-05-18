import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import pickle

def model(df, method):
    corr_df = df.corr(method=method)
    return corr_df

termFrequency_df = pickle.load(open('assets/termFrequencyModel.p', 'rb'))
termSemantic_df = pickle.load(open('assets/termSemanticModel.p', 'rb'))

all_df = pd.merge(left=termFrequency_df, right=termSemantic_df, on='Term', how='inner')
all_df = all_df[(all_df['Count:Year:2020'] > 100)]  #We only care term that has more than 100 count
pickle.dump(all_df, open('termFreqSemantic.p', 'wb')) 

corr_df = model(all_df, 'pearson')
pickle.dump(corr_df, open("correlation_pearson.p", "wb" ))

corr_df = model(all_df, 'spearman')
pickle.dump(corr_df, open("correlation_spearman.p", "wb" ))