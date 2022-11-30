import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.svm import SVR
from itertools import product
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


class Minesweeper:
    def __init__(self, parameter_space):
        self.b = 3
        self.df = pd.DataFrame(columns=parameter_space.keys())
        self.df['score'] = []
        self.mms = MMS().fit(pd.DataFrame(parameter_space))
        self.dimensions = len(parameter_space)
        
    @property
    def bins(self):
        return np.linspace(0,1,self.b+1)
    
    @property
    def all_cells(self):
        return set(product(range(self.b), repeat=self.dimensions))
    
    @property
    def occupied_cells(self):
        return list(set(self.df.apply(lambda x:np.digitize(x, self.bins)-1,
                                     result_type='reduce', axis=1).apply(tuple)))
    @property
    def exclude_cells(self):
        if len(self.df)==0: return set([])
        exclude_cells = list(set([(tuple([i+j for i,j in zip(a,b)])) 
                                      for b in self.relative_neighbors for a in self.occupied_cells]))
        exclude_cells = set([x for x in exclude_cells if all(0<=i<self.b for i in x)])
        return exclude_cells
            
    @property
    def relative_neighbors(self):
        result = np.array(list(product((-1,0,1), repeat=self.dimensions)))
        result = result[np.linalg.norm(result, axis=1)<(self.dimensions)**.5]
        return result
    
    def get_parameters(self):
        scores = self.df.pop('score')
        while True:        
            vacant_cells = list(self.all_cells-self.exclude_cells)
            if len(vacant_cells)==0:
                self.b+=1
                print(f'increasing b to {self.b}')
                continue
                
            chosen_cell = vacant_cells[np.random.randint(0,len(vacant_cells))]
            chosen_values = tuple([(self.bins[i]+self.bins[i+1])/2 for i in chosen_cell])
            self.df['score']=scores
            return {k:v for k,v in zip(self.mms.feature_names_in_,
                                       self.mms.inverse_transform([chosen_values])[0])}
        
    def update_values(self,values,score):
        values = pd.DataFrame(values, index=[0])
        values = pd.DataFrame(data=self.mms.transform(values), columns=values.columns)
        values['score'] = score
        
        self.df = pd.concat([self.df, values], ignore_index=True).reset_index(drop=True)
    
    @property
    def history(self):
        result = pd.DataFrame(data=self.mms.inverse_transform(self.df.iloc[:,:-1]), columns=self.mms.feature_names_in_)
        result['score'] = self.df['score']
        return result