# %%
import pandas as pd
import numpy as np
import sys
import intertools

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

data = data.dropna()

features = data.columns()

#var_ranking1 = var_ranking1.reset_index()
rm_features = np.array(['ASSOC_PROB_LR'])
for i in range(0,num):
    rm_features = np.append(rm_features,var_ranking1['features'][i])

for i in range(0,rm_num):
    for j in 
    remove_features = itertools.combinations()
    reduced_dataset['column_' + str(j)]



newdata = data.drop(columns=rm_features)

# %%
newdata.to_csv('Fermi-LAT Data\\fl_varranked_rm' + str(num) + '.csv')


