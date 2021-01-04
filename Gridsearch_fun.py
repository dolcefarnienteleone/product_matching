import numpy as np

class RF_GridSearch():

    def creat_grid(self):
        ## Create Grid for RF
        # no. of trees in RF
        n_estimators = [int(x) for x in np.linspace(200, 1000, 5)]

        # Max no. of levels in tree
        max_depth = [int(x) for x in np.linspace(5, 55, 11)]
        # Add default value
        max_depth.append(None)

        # No. of features to consider at every split
        max_features = ['sqrt', 'log2']

        # Criterion to split on
        criterion = ['gini', 'entropy']

        # To split a node, the Min no. of samples required
        min_samples_split = [int(x) for x in np.linspace(2, 10, 9)]

        # Min decrease in impurity required for split to happen
        min_impurity_decrease = [0.0, 0.05, 0.1]

        # Method of selecting samples for training each tree
        bootstrap_rule = [True, False]

        ###Create the grid
        grid = {'n_estimators': n_estimators,
                'max_depth': max_depth,
                'max_features': max_features,
                'criterion': criterion,
                'min_samples_split': min_samples_split,
                'min_impurity_decrease': min_impurity_decrease,
                'bootstrap': bootstrap_rule}

        return grid
        #print(f'The grid search paras:{grid}')


    def get_bestparam(self, X, y):
        ###initialize the RF model
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.ensemble import RandomForestClassifier
        RF = RandomForestClassifier()
        grid = self.creat_grid()
        RF_rdsearch = RandomizedSearchCV(estimator=RF, param_distributions=grid,
                                         n_iter=200, cv=3, verbose=2, random_state=42,
                                         n_jobs=-1)

        # Fit the random search model
        RF_rdsearch.fit(X, y)

        # View the best parameters from the random search
        #print(f'best_para = {RF_rdsearch.best_params_}')
        return RF_rdsearch.best_params_

class MLP_GridSearch():

    def creat_grid(self):
        hidden_layer_sizes = [(100,), (300,), (500,)]
        activation = ['logistic', 'tanh', 'relu']
        solver = ['sgd', 'adam']
        early_stopping_rule = [True, False]

        NN_grid = {'hidden_layer_sizes': hidden_layer_sizes,
                   'activation': activation,
                   'solver': solver,
                   'early_stopping': early_stopping_rule,
                   }

        return NN_grid
        #print(f'The grid search paras:{NN_grid}')

    def get_bestparam(self, X, y):
        ###initialize the RF model
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.neural_network import MLPClassifier
        NN = MLPClassifier()
        NN_grid = self.creat_grid()
        NN_random = RandomizedSearchCV(estimator=NN, param_distributions=NN_grid,
                                       n_iter=200, cv=3, verbose=2, random_state=31,
                                       n_jobs=-1)
        NN_random.fit(X, y)

        return NN_random.best_params_
        # View the best parameters from the random search
        #print(f'best_para = {NN_random.best_params_}')