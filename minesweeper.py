'''TODO:

    make .values_to_parameters() method
    cache last suggested parameters for fast updates
'''


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler as MMS
from itertools import product





class Minesweeper:
    """Minesweeper: A tool for simplified grid search over parameter space.

    Minesweeper is a simplifed tool for performing a grid search over a parameter
    space in any number of dimensions. Minesweeper maintains a dataframe of parameter
    sets that have already been used and suggests new combinations of parameters from
    a blank region of parameter space.

    Unlike other GridSearch tools, Minesweeper does not do additional functions
    such as construct, train or score models, perform N-fold cross validation, or use
    callbacks. These functions can more easily and more flexibily be acheived outside
    of the grid search tool. This also makes it easy to interrupt and continue the
    tuning process.

    Minesweeper efficiently covers the parameter space by using a series of progressively
    smaller grids to look for blank spaces on the parameter map to explore. It is 
    invariant to scale and cardinality. I.e, it can search over a range of [0...1000] as
    efficiently as a range of [0..1].

    The search over parameter space is unordered so it does not repeatedly draw from one
    area of the space. The search is exhaustive unless one or more parameters is inexhaustabile 
    (i.e. a float). Minesweeper will terminate by returning `None` if 
    all combinations of parameters have been used.








    """
    def __init__(self, parameter_space, verbose=False):
        self.b = 3
        self.df = pd.DataFrame(columns=parameter_space.keys())
        self.casting_functions = {k:v['func'] if 'func' in v else float if max(v['range'])==1 else int for k,v in parameter_space.items()}
        self.df['score'] = []
        self.mms = MMS().fit(pd.DataFrame({k:v['range'] for k,v in parameter_space.items()}))
        self.dimensions = len(parameter_space)
        self.previous_values = {}
        self.verbose = verbose
        
    @property
    def bins(self):
        return np.linspace(0,1,self.b+1)
    
    @property
    def all_cells(self):
        return set(product(range(self.b), repeat=self.dimensions))
    
    @property
    def occupied_cells(self):
        return list(set(self.df.iloc[:,:-1].apply(lambda x:np.digitize(x, self.bins)-1,
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


    @property
    def vacant_cells(self):
        vacant_cells = list(self.all_cells - self.exclude_cells)
        # remove any vacant cells that map to parameters in self.previous_values
        collector = []
        for cell in vacant_cells:
            cell_center = tuple([(self.bins[i]+self.bins[i+1])/2 for i in cell])
            values = {k:v for k,v in zip(self.mms.feature_names_in_, cell_center)}
            parameters = {k:self.casting_functions[k](v) for k,v in zip(self.mms.feature_names_in_, self.mms.inverse_transform([cell_center])[0])}
            if str(parameters) in self.previous_values:
                if self.verbose: print('Already done')
                self.previous_values[str(parameters)]['values'] += [values]
                self.update_values(parameters, score=self.previous_values[str(parameters)]['score'])
            else:
                collector.append(cell)
        return collector
    
    def get_parameters(self):
        attempts = 0
        while True:        
            vacant_cells = self.vacant_cells
            if len(vacant_cells)==0:
                self.b+=1
                attempts +=1
                if attempts>5:
                    return None

                if self.verbose: print(f'increasing b to {self.b}')
                continue
                
            chosen_cell = vacant_cells[np.random.randint(0,len(vacant_cells))]
            chosen_cell_center = tuple([(self.bins[i]+self.bins[i+1])/2 for i in chosen_cell])

            values = {k:v for k,v in zip(self.mms.feature_names_in_, chosen_cell_center)}
            parameters = {k:self.casting_functions[k](v) 
                for k,v in zip(self.mms.feature_names_in_, self.mms.inverse_transform([chosen_cell_center])[0] )}

            self.previous_values[str(parameters)] = {'values':[values]}
            return parameters


    def update_values(self, parameters, score):
        self.previous_values[str(parameters)]['score'] = score
        values = self.previous_values[str(parameters)]['values'][-1]
        values = pd.DataFrame(values, index=[0])
        values['score'] = score
        self.df = pd.concat([self.df, values], ignore_index=True).reset_index(drop=True)
        
    
    @property
    def history(self):
        if len(self.df)==0:
            return pd.DataFrame(columns=self.mms.feature_names_in_+['score'])

        return pd.DataFrame([eval(k)|{'score':v['score']} for k,v in self.previous_values.items()])


