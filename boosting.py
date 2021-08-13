import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from decision_tree import DT

class BoostingTree: 
    def __init__(self,
                 mode,
                 data,
                 method = 'average',
                 num_trees= 3,
                 max_depth= 1,
                 learning_rate=0.1):
        self.data = data
        self.trees = []
        self.num_trees = num_trees
        self.mode = mode
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.method = method
        
    def get_base_pred(self, data): 
        if self.mode == 'regression':
            base_pred = np.mean(data.iloc[:, -1])
        else: 
            class_1 = self.data[self.data['Label'] == 1].shape[0]
            class_0 = self.data[self.data['Label'] == 0].shape[0] 

            base_pred = np.log(class_1 / class_0)
        return base_pred
    
    def sigmoid(self, data): 
        return np.exp(data) / (1 + np.exp(data))
    
    def cost(self, y, p):
        if self.mode == 'regression':
            cost = np.sum((y - p) ** 2) / y.shape[0]
        else: 
            cost = 0
            epsilon =   2.22e-16
            cost = np.mean(- ( y * np.log(p + epsilon) + (1 - y) * np.log( 1 - p + epsilon)))
            
        return cost
    
    def predict(self):
        pred = self.get_base_pred(self.data)
        base_pred = pred
        odds = pred - (1- pred)
        new_data = self.data.copy()
        
        for i in range(self.num_trees):
            if self.mode == 'regression':
                result = 0
                new_data.iloc[:, -1] = new_data.iloc[:, -1] - pred
                dt = DT(mode='regression', data=new_data, max_depth=self.max_depth, method=self.method)
                dt.fit()
                
                pred = dt.predict(new_data)
                result *= self.learning_rate * pred

                self.trees.append(dt)
            else: 
                new_data.iloc[:, -1] = new_data.iloc[:, -1] - pred
                dt = DT(mode='regression', data=new_data, max_depth=self.max_depth, method='non_average')
                dt.fit()
                
                pred_raw = dt.predict(new_data)
                result = []
                
                for p_r in pred_raw: 
                    log_odds_val = np.sum(p_r) / (p_r).shape[0] * (odds)
                    result.append(self.sigmoid(base_pred + self.learning_rate * log_odds_val))
                
                result = np.array(result)
                pred = result
        
        if self.mode == 'classification':
            result = np.where(result > 0.5, 1, 0)
            
        return result        