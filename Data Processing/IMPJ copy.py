import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class DataProcessor:
    def __init__(self):
        pass

    def choose_2_vars(self,classname,data,A,B):
        '''
        In 'data' all unknown BCUs are assigned 0. 
        All BLLs are assigned 1 and all FSRQs are assigned 2.
        Radio Galaxies = 3 and Other = 4

        The function will mask all values except the 2 desired classes
        '''
        for x in data[classname].unique():
            if x == A or x == B:
                pass
            else:
                mask = data[classname] == x
                data = data[~mask]

        Y = np.zeros(len(data[classname]))

        for i in range(0,len(data[classname])):
            if np.array(data[classname])[i] == A:
                pass
            if np.array(data[classname])[i] == B:
                Y[i] = 1
        print('check1')
        return data.loc[:, data.columns != classname] , Y
    
    def logtransform(self,data,features):
        
        featurelist = features
        for x in featurelist:
            if np.min(data[x]) <=0:
                data[x] = np.log(data[x] + 1 + abs(np.min(data[x])))
            else:
                data[x] = np.log(data[x])
        return data
    
    def multipleimputation(self,data,col,rmindices):
        values = rmindices
        ls = []
        for x in values:
            ls.append(data.loc[[x],[col]])
            data.loc[[x],[col]]=np.nan
        print(ls)
        ls=[]
        for x in values:
            ls.append(data.loc[[x],[col]])
        print(ls)
        missing_mask = ~data.isna()

        imputer = IterativeImputer(max_iter=1000,random_state=23,verbose=2)
        imputed_values = imputer.fit_transform(data)

        data = data.where(missing_mask,imputed_values)
        ls=[]
        for x in values:
            ls.append(data.loc[[x],[col]])
        print(ls)
        return data