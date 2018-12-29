# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 14:33:48 2018

@author: Shawlock
"""
#Let's check the label and length of the data
personality.keys()        
len(personality['posts'])

#We take a look to the classes. It looks like it is a very unbalanced dataset:

cnt_srs = personality['type'].value_counts()

plt.figure(figsize=(12,4))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Types', fontsize=12)
plt.show()

#words per comment and Varience of word_count
def var_row(row):
    l = []
    for i in row.split('|||'):
        l.append(len(i.split()))
    return np.var(l)

personality['words_per_comment'] = personality['posts'].apply(lambda x: len(x.split())/50)
personality['variance_of_word_counts'] = personality['posts'].apply(lambda x: var_row(x))
personality.head()

#words_per_comment according to the Type
plt.figure(figsize=(15,10))
sns.swarmplot("type", "words_per_comment", data=personality)