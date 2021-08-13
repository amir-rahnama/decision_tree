import numpy as np
import pandas as pd
import graphviz


class Node:
    def __init__(self,
                 node_id,
                 split_criterion,
                 true_branch,
                 false_branch,
                 depth,
                 gain,
                 data):
        self.node_id = node_id
        self.split_criterion = split_criterion
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.depth = depth
        self.gain = gain
        self.data = data

class DT: 
    def __init__(self,
                 mode,
                 data,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_depth=None,
                 method='average',
                 class_names=None):
        self.data = data
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.depth = 1
        self.nodes = []
        self.base_node_id = 0
        self.mode = mode
        self.method = method
        self.class_names = class_names

    def gini(self, data):
        if data.shape[0] == 0: 
            return 0
        counts = self.class_counts(data)
        val = np.array(list(counts.values()))
        portions = val / len(data)
        return 1 - np.sum(portions**2)
    
    def mse(self, data): 
        y = data.iloc[:, -1]
        avg_pred = np.mean(y)
        return np.sum((y - avg_pred)**2)
    
    def cost(self, data): 
        if self.mode == 'classification':
            return self.gini(data)
        else: 
            return self.mse(data)
    
    
    def info_gain(self, true_rows, false_rows, current_cost):
        p = round(len(true_rows) / (len(true_rows) + len(false_rows)), 3)
        return current_cost - p * self.cost(true_rows) - (1 - p) * self.cost(false_rows)
    
    def partition(self, data, feature, val):
        total_rows= data.shape[0]
        
        if data.dtypes[feature] == 'O':
            condition =  data.iloc[:, feature] == val
        else: 
            condition =  data.iloc[:, feature] >= val
        
        true_idx = np.argwhere(condition.values).ravel()
        false_idx = np.setxor1d(np.arange(total_rows), true_idx)
        
        return  data.iloc[true_idx,:], data.iloc[false_idx,:]
    
    def find_split(self, data):
        best_gain = 0
        current_cost = self.cost(data)
        num_features = data.shape[1] - 1

        for n_f in range(num_features):
            unique_vals = np.unique(data.iloc[:, n_f])
            for uniq_val in unique_vals:
                filter_criteria = {'feature': n_f, 'value': uniq_val}

                true_rows, false_rows = self.partition(data, n_f, uniq_val)

                gain = self.info_gain(true_rows, false_rows, current_cost)
                if gain >= best_gain:
                    best_gain, best_filter = gain, filter_criteria
        
        return best_gain, best_filter
    
    def counter(self):
        self.base_node_id += 1
        return self.base_node_id
    
    def grow_tree(self, data, node_id):
        best_gain, best_filter = self.find_split(data)

        true_rows, false_rows = self.partition(data, best_filter['feature'], best_filter['value'])    
        num_samples = len(true_rows) + len(false_rows)
        is_leaf = (best_gain == 0 or num_samples <= self.min_samples_leaf) or \
                  (self.max_depth is not None and self.depth > self.max_depth)
        
        if is_leaf : 
            self.nodes.append(Node(node_id, best_filter, None, None, self.depth, best_gain, data))
        else:
            right_node_id = self.counter()
            left_node_id = self.counter()

            self.nodes.append(Node(node_id, best_filter, right_node_id, left_node_id, self.depth, best_gain, data))

            self.depth = self.depth + 1
            self.grow_tree(true_rows, right_node_id)
            self.grow_tree(false_rows, left_node_id)        

    def fit(self):
        node_id = self.counter()
        return self.grow_tree(self.data, node_id)
    
    
    def leaf_pred(self, data, node):
        if self.mode == 'classification':
            cc = self.class_counts(node.data)
            cc_val = np.array(list(cc.values()))

            cc_name = np.array(list(cc.keys()))
            sum_cc_val = np.sum(cc_val)

            pz = np.zeros(len(class_names))
            for i in range(len(cc_name)): 
                c_idx = np.argwhere(cc_name[i] == class_names)[0][0]
                pz[i] = cc_val[i] / sum_cc_val

            return pz
        else: 
            if self.method == 'average': 
                result =  np.mean(node.data.iloc[:,-1])
            else: 
                result = node.data.iloc[:,-1].values
            return result
    
    def predict(self, data):
        pred_result = []
        for i in range(data.shape[0]):
            pred_result.append(self.pred(data.iloc[i]))
        return np.array(pred_result)

        
    def pred(self, data, node_id=1):
        node = self.find_node(node_id)

        if not node.true_branch:      
            return self.leaf_pred(data, node)

        is_true = self.is_condition_true(node.split_criterion, data)

        if (is_true):
            next_node = node.true_branch
        else:
            next_node = node.false_branch

        return self.pred(data, next_node)

    def draw(self):
        dot = graphviz.Digraph()
        
        for i in range(len(self.nodes)):
            if self.nodes[i].true_branch: 
                dot.node(str(self.nodes[i].node_id), self.get_criteria_strig(self.nodes[i].split_criterion))
            else: 
                if self.mode == 'classification':
                    dot.node(str(self.nodes[i].node_id), str(self.class_counts(self.nodes[i].data)))
                else: 
                    dot.node(str(self.nodes[i].node_id), str(np.mean(self.nodes[i].data.iloc[:, -1])))
        edges = []
        for i in range(len(self.nodes)):
            node_id = str(self.nodes[i].node_id)
            right_node_id = str(self.nodes[i].true_branch)
            left_node_id = str(self.nodes[i].false_branch)
            
            if right_node_id != 'None':
                dot.edge(node_id, right_node_id, label='TRUE')
                dot.edge(node_id, left_node_id, label='FALSE')
        
        return dot
    def find_node(self, node_id):
        for i in range(len(self.nodes)):
            if node_id == self.nodes[i].node_id:
                return self.nodes[i]  
            
    def get_criteria_strig(self, split_criterion):
        symbol = '==' if self.data.dtypes[split_criterion['feature']] == 'O' else '>='
        return 'X[{}] {} {}'.format(split_criterion['feature'], symbol, split_criterion['value'])
    
    def is_condition_true(self, split_criterion, data): 
        if self.data.dtypes[split_criterion['feature']] == 'O':
            condition_val = data[split_criterion['feature']] == split_criterion['value']
        else: 
            condition_val = data[split_criterion['feature']] >= split_criterion['value']
        return condition_val
    
    def class_counts(self, data):
        class_idx = 2
        res = {}
        for c_name in self.class_names: 
            count = np.sum(data['label'] == c_name)
            res[c_name] = count
        return res