# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 01:20:39 2019

@author: michael

hockey stats
"""


#read in data
#check empty cols, ranges, values

#if player appears more than once, lets just use the TOT

#can we cluster and visualize to see groups of "types" of players
#can we differentiate Forwards from Defense (maybe C, LW, RW?) ie predict
# different models same data for predict

#splitting by position, are the clusters interesting?


import pandas as pd

df = pd.read_csv('data\skaterbasicstats18.csv', header=1)

df.isnull().sum()
#shows "S%" and "FO%" have null values, make those 0

spercent = df["S%"]
fopercent = df["FO%"]

spercent.fillna(0, inplace=True)
fopercent.fillna(0, inplace=True)

df.isnull().sum()
#now we have no nulls


#next step, combine duplicate, use only first, which should be total

df[df.duplicated(['Player'])]
#shows we have duplicate players
df.drop_duplicates(['Player'], inplace=True)
#now no duplicates



#misc visualizations
df.boxplot(column='PTS', by='Pos')

df.boxplot(column='+/-', by='Pos')






#PCA and visualization
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


cols = list(df.columns)
cols.remove('ATOI')
#Average Time on Ice is a linear combination of Games Played and Time
#   on Ice, so is redundant.
features = cols[5:]


x = df.loc[:,features].values
y = df.loc[:,['Pos']].values
X = StandardScaler().fit_transform(x)

pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(X)
principalDF = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])

target = pd.DataFrame(y,columns=['Pos'])

finalDF = pd.concat([principalDF,target], axis=1)

def offdef(x):
    if x in ['LW','RW','C','W']:
        return 'F'
    else:
        return 'D'
#right now just focus on Forward/Defense        
finalDF['target'] = finalDF['Pos'].apply(offdef)

fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(1,1,1)
targets = ['F','D']
colors = ['r','b']

for target, color in zip(targets, colors):
    indicesToKeep = finalDF['target'] == target
    ax.scatter(finalDF.loc[indicesToKeep, 'pc1'],
               finalDF.loc[indicesToKeep, 'pc2'],
                c = color,
                s = 50)



#can we predict Forward vs. Defense from basic stats?
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=1/7.0)


