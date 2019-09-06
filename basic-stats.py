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

df.boxplot(column='PTS', by='Pos')