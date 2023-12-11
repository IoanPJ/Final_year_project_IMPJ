import numpy as np
import pandas as pd

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
    
    def logtransform(self,data):
        
        featurelist = ['Pivot_Energy']

        for x in featurelist:
            data[x] = np.log(data[x])
        return data