import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df=pd.read_csv('TR.csv')

df.head()

#sns.pairplot(df[['ACS','ADR','APR','FKPR','KPR','FDPR','K:D']]) #fkpr #APR

import sklearn.cluster as cluster

kmeans = cluster.KMeans(n_clusters=2, init='k-means++')
kmeans = kmeans.fit(df[['FKPR','APR']])

df['cluster'] = kmeans.labels_

df.head()


sns.scatterplot(x='FKPR', y='APR', hue='cluster', data=df)

plt.savefig('kmean.pdf')
