import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NearestCentroid, NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.utils.extmath import randomized_svd
from numpy import linalg as la
from scipy.sparse import csr_matrix

df = pd.read_csv('./Crime_Data_Selected_Cleaned.csv',sep=",",on_bad_lines="skip",encoding='latin-1')

df_tranform = df.copy()
df_tranform['DATE OCC'] = pd.to_datetime(df_tranform['DATE OCC'], format='%m/%d/%Y %H:%M')
df_tranform['Year'] = df_tranform['DATE OCC'].dt.year
df_tranform['Month'] = df_tranform['DATE OCC'].dt.month
df_tranform['Day'] = df_tranform['DATE OCC'].dt.day
df_tranform['DayOfWeek'] = df_tranform['DATE OCC'].dt.day_name()
df_tranform.drop('DATE OCC', axis=1, inplace=True)

df_tranform['TIME OCC'] = df_tranform['TIME OCC'].astype(str).str.zfill(4)
df_tranform['TIME OCC'] = pd.to_datetime(df_tranform['TIME OCC'], format='%H%M').dt.time


unique_counts = df_tranform[['AREA NAME', 'Premis Desc', 'Crm Cd Desc']].nunique()

print(unique_counts)


cols = ['AREA NAME', 'Premis Desc', 'Crm Cd Desc']
df_subset = df_tranform[cols].dropna()
transactions = df_subset.apply(lambda x: list(x), axis=1).tolist()

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(np.asarray(te_ary), columns=te.columns_)


print("\n\n")


frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])