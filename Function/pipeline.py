''' A py file to def function'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, accuracy_score,precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold  
#--------------------- Applying Weight of Evidence and Information Value (IV)----------------------------------
class WOE: 
    def __init__(self):
        self.result_dict = {}
        self.filtered_cols = []

    # Defining threshold values for IVs
    def _interpret_iv(self, iv_value):
        if iv_value < 0.02:
            return 'Does not appear to be useful for prediction'
        elif iv_value < 0.1:
            return 'Weak predictive power'
        elif iv_value < 0.3:
            return 'Medium predictive power'
        else:
            return 'Strong predictive power'


    def _calculate_WOE(self, data, feature, target, nums=0, num_bins = 10):
        ''' 
        Calculate total number of positive and negative instances
        num: check if feature is numeric or not, 0: no, 1: yes
        '''
        df = data.copy()
        if nums == 1:
            df[feature] = pd.qcut(df[feature], num_bins, duplicates= 'drop')
        grouped = df.groupby(feature)[target].agg(['count', 'sum'])
        grouped = grouped.rename(columns={'count': 'total_count', 'sum': 'Default_counts'})
        grouped['Non_Default_counts'] = grouped['total_count'] - grouped['Default_counts']

        # Calculate WOE and IV (Information Value)
        grouped['Distr_non_Default'] = grouped['Non_Default_counts'] / grouped['Non_Default_counts'].sum()
        grouped['Distr_Default'] = grouped['Default_counts'] / grouped['Default_counts'].sum()
        grouped['WoE'] = np.log(grouped['Distr_non_Default'] / grouped['Distr_Default'])
        grouped = grouped.replace({'WoE': {np.inf: 0, -np.inf: 0}})
        grouped['IV'] = (grouped['Distr_non_Default'] - grouped['Distr_Default']) * grouped['WoE']
        grouped['Features'] = feature
        grouped['Type'] = nums #if features is num = 1, else = 0

        return grouped

    def _return_woe_data(self, df, target, bin = 10):
        woe_list = []
        for col in df.columns:

            if col == 'Target':
                continue
            elif df[col].dtype == 'object':
                woe_list.append(self._calculate_WOE(df, col, target))
            else:
                woe_list.append(self._calculate_WOE(df, col, target, nums= 1, num_bins= bin))

        result = pd.concat(woe_list)
        df_iv = result.groupby('Features')[['IV']].sum()
        df_iv['Interpretation'] = df_iv['IV'].apply(self._interpret_iv)
        return result.reset_index(), df_iv.reset_index()


    def fit(self, x_train, y_train):
        data = pd.concat([x_train, y_train], axis= 1)
        woe_result, df_iv = self._return_woe_data(data, y_train.name)
        self.filtered_cols = df_iv[df_iv['Interpretation'] != 'Does not appear to be useful for prediction']['Features'].values

        filter_woe_result = woe_result[woe_result['Features'].isin(self.filtered_cols)]
        filter_woe_result = filter_woe_result[['Features','index','WoE','Type']]
        
        # make the mapping dictionary
        for index, row in filter_woe_result.iterrows():
            feature = row['Features']
            bins = row['index']
            woe = row['WoE']
            if feature not in self.result_dict:
                self.result_dict[feature] = {}
            self.result_dict[feature][bins] = woe
        # Mapping
        for key in self.result_dict.keys():
            data[key] = data[key].map(self.result_dict[key])

        return data[[col for col in self.filtered_cols]], df_iv, woe_result
    
    def transform(self, x_test):
        data = x_test.copy()
        for key in self.result_dict.keys():
            data[key] = data[key].map(self.result_dict[key])
        data = data[[col for col in self.filtered_cols]] 
    
        ''' missing values still occurs in data'''
        missing_cols = data.columns[data.isnull().any()]
        for col in missing_cols:
            replacement_value = sum(self.result_dict[col].values()) / len(self.result_dict[col]) 
            data[col].fillna(replacement_value, inplace=True)
        return data
        






#-----------------------------Remvoving Outliers----------------------------------------------------------------

def calculate_z_scores(data,column):
    
    mean = data[column].mean()
    std_dev = data[column].std()
    # Calculate Z-scores for each data point in the column
    z_scores = (data[column] - mean) / std_dev
    
    return z_scores

def trimming(data, col, threshold):
    '''Remove the outliers with z_score >= threshold'''
    df = data[col]
    z_scores = calculate_z_scores(data,col)
    df_filtered = df[z_scores < threshold]
    return data.loc[df_filtered.index]

def winsorization(data, col, threshold, replacement_value):
    '''Replace the outliers with z_score >= threshold with replacement_value'''
    z_scores = calculate_z_scores(data, col)
    # Replace outliers with replacement_value
    data[col] = np.where(z_scores >= threshold, replacement_value, data[col])
    return data

#-----------------------------Split data----------------------------------------------------------------
def split_train_test(data, target_col, percent):
    X = data.drop(columns=[target_col])  
    y = data[target_col]
    # Splitting the dataset into training and testing sets (adjust test_size as needed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= percent, random_state=42)
    return X_train,X_test,y_train,y_test



#-----------------------------Select Kbest----------------------------------------------------------------

def selectkbest(X,y , percent_keep):
    '''
    X : input dataframe
    percent_keep : the percentage of features chosen 
    '''
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif
    selector = SelectKBest(score_func=f_classif, k= int(percent_keep * len(X.columns)))  
    X.columns = X.columns.astype(str)
    # Fit the selector to your data
    X_new = selector.fit_transform(X, y)

    # Get the selected feature indices
    selected_features_indices = selector.get_support(indices=True)
    # Get the names of the selected features
    selected_feature_names = X.columns[selected_features_indices]
    X = X[selected_feature_names]

    return pd.concat([X,y],axis= 1)


#-----------------------------Predict_Model----------------------------------------------------------------
def prediction_result(model, X_train,X_test,y_train,y_test):
    '''predict data'''
    # Fit the model
    model.fit(X_train, y_train) 

    # Predict class labels
    y_pred = model.predict(X_test)
    # Predict probabilities
    y_proba = model.predict_proba(X_test)[:, 1]

    # prob dataframe
    PD = pd.DataFrame({'Probability of Default': y_proba, 'Prediction': y_pred, 'Actual':y_test})

    return PD



#-----------------------------Cross Validation----------------------------------------------------------------

def cross_val(model, nums_fold, X, y, print_ = 1):
    '''Cross validation for train data to check if the model is good or not '''
    import time
    start_time = time.time()
    kf = StratifiedKFold(n_splits= nums_fold, shuffle=True, random_state=42)

    cross_val_scores = cross_val_score(model, X, y, cv=kf)
    end_time = time.time()
    if print_ == 1:
        print(f'{model} - With {nums_fold} folds - Finsh in {end_time - start_time} seconds')
        print("Cross-validation scores:", cross_val_scores)
        print("Mean CV score:", cross_val_scores.mean())
    
    return cross_val_scores.mean()

class ModelEvaluation:
    
    """
    A class to compute the auc score & gini of the model for the given predictions.

    Attributes
    ----------
    predictions : pd.DataFrame
        A DataFrame containing 'Probability_Default', Prediction and Actual columns.
        
    """
    
    def __init__(self, predictions_df,):
        self.predictions = predictions_df

    def plot_roc_curve(self):
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        # Compute FPR, TPR, and thresholds
        fpr, tpr, thresholds = roc_curve(y_true = self.predictions['Actual'], y_score = self.predictions['Probability of Default'])
        # Compute AUC
        roc_auc = auc(fpr,tpr)
        # Plot the ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (AUC = %0.5f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'r--')  # Plotting the diagonal line (random classifier)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
    
    
    def compute_auc_gini(self):
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_score = self.predictions['Probability of Default'],y_true = self.predictions['Actual'])
        gini = 2*auc - 1
        return auc,gini
    
    def compute_conf_mat(self):
        cm = confusion_matrix(self.predictions['Actual'], self.predictions['Prediction'])
        acc = accuracy_score(self.predictions['Actual'], self.predictions['Prediction'])
        precision, recall, f1_score, _ = precision_recall_fscore_support(self.predictions['Actual'], self.predictions['Prediction'], average='binary')
        
        return cm, acc, precision, recall, f1_score

