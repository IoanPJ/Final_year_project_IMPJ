# %%
import pandas as pd
import numpy as np
import sys

num = int(sys.argv[1])


data = pd.read_csv('Fermi-LAT Data\\fl_numericalonly_nopositional_withclasses.csv',index_col=0)

'''
In 'data' all unknown BCUs are assigned 0. 
All BLLs are assigned 1 and all FSRQs are assigned 2.
Radio Galaxies = 3 and Other = 4
'''

mask1 = data['CLASS1'] == 4
mask2 = data['CLASS1'] == 0
mask3 = data['CLASS1'] == 3
data = data[~mask1]
data = data[~mask2]
data = data[~mask3]

''' JOINING BLLs AND FSRQs INTO A SINGLE CATEGORY '''

#data = data.replace(2,1)

data = data.dropna()
#data = data.reset_index()

#print(len(data['CLASS1']))
#data_test = data[~(data['CLASS1'] == 1)]
#print(len(data_test['CLASS1']))
#print(data.columns)

# %% [markdown]
# We will split the dataset (obtained from sklearn load_digits) into a training and test set using the code below 

# %%

X = data.loc[:, data.columns != 'CLASS1']
Y = data['CLASS1']


# %%
from skfeature.function.similarity_based import fisher_score
import matplotlib.pyplot as plt

ranks = fisher_score.fisher_score(np.array(X),np.array(Y))

#print(ranks)
#print(data.columns)
#print(ranks,np.array(data.columns))
feat_importances = pd.DataFrame((np.array([ranks, np.array(X.columns)]).T),columns=['ranks','features'])



#fisher = pd.DataFrame((feat_importances),columns=['features','fisher score'])
#feat_importances.head

# %%
from sklearn.feature_selection import r_regression

ranks = r_regression(X,Y)

''' 
Pearson's 'r' calculated as E[(X[:, i] - mean(X[:, i])) * (y - mean(y))] / (std(X[:, i]) * std(y))
'''


pearson = pd.DataFrame((np.array([ranks, np.array(X.columns)]).T),columns=['pearson','features'])
var_ranking = pd.merge(pearson,feat_importances, on = 'features')
var_ranking = var_ranking.sort_values(by='ranks')
#print(var_ranking)


# %%
from sklearn.feature_selection import mutual_info_classif

mi = mutual_info_classif(X,Y)


''' 
Mutual info classification for each variable
'''


mutual = pd.DataFrame((np.array([mi, np.array(X.columns)]).T),columns=['mutual information','features'])
var_ranking1 = pd.merge(mutual,var_ranking, on = 'features')
var_ranking1 = var_ranking1.sort_values(by='mutual information')
#print(var_ranking)



# %%

var_ranking1 = var_ranking1.sort_values(by='mutual information')
mi_rank = np.arange(0,len(var_ranking1['mutual information']),1)
var_ranking1['mutual information score'] = mi_rank
var_ranking1 = var_ranking1.sort_values(by='pearson')
pearson_score = np.arange(0,len(var_ranking1['pearson']),1)
var_ranking1['pearson score'] = pearson_score
var_ranking1['total score'] = var_ranking1['pearson score'] + var_ranking1['mutual information score']

var_ranking1 = var_ranking1.sort_values(by='total score',ascending=True)

#var_ranking1 = var_ranking1.reset_index()
rm_features = np.array(['ASSOC_PROB_LR'])
for i in range(0,num):
    rm_features = np.append(rm_features,var_ranking1['features'][i])

newdata = data.drop(columns=rm_features)

# %%
newdata.to_csv('Fermi-LAT Data\\fl_varranked_rm' + str(num) + '.csv')


