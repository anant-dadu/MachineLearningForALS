import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import shap
import logging
import lightgbm as lgb
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import pickle 
from sklearn.metrics import roc_auc_score

from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import copy


logging.basicConfig(
    level=logging.INFO,
    # format="%(asctime)s [%(levelname)s] %(message)s",
    format='%(asctime)s %(message)s', 
    handlers=[
        logging.StreamHandler()
    ]
)

logger=logging.getLogger() 
logger.setLevel(logging.INFO) 
# logger.debug("Harmless debug Message") 
# logger.info("Just an information") 
# logger.warning("Its a Warning") 
# logger.error("Did you try to divide by zero") 
# logger.critical("Internet is down") 

def calculate_roc_auc_score_multiclass(y_pred, y_true, type_true='1d'):
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    if type_true == '1d':
        mlb = MultiLabelBinarizer()
        y_true = mlb.fit_transform([(str(int(i))) for i in y_true])
    auc = []
    for c in range(y_pred.shape[1]):
        auc.append(metrics.roc_auc_score(y_true[:, c], y_pred[:, c]))
    micro_auc_score = metrics.roc_auc_score(y_true, y_pred, average='micro')
    return auc, micro_auc_score


class generateFullData:

    def __init__(self):
        pass
    
    def trainXGBModel_binaryclass(self, data, feature_names, label_name, replication_set=None):
        logger.info('Training starts...')
        X_train, X_valid, y_train, y_valid = train_test_split(data, data[label_name] , test_size=0.3, random_state=42)
        if replication_set is not None:
            X_train = data.copy()
            y_train = data[label_name].copy()
            replication_dataframe = replication_set[feature_names].copy()
            y_rep_true = replication_set[label_name] 
            X_valid = replication_set.copy()
            y_valid = y_rep_true.copy()
            drep = xgb.DMatrix(replication_dataframe, label=y_rep_true, feature_names=feature_names)

        ID_train = X_train['ID']
        ID_test = X_valid['ID']
        X_train = X_train[feature_names].copy()
        X_valid = X_valid[feature_names].copy()
        num_round = 500
        param = {
            'objective':'binary:logistic',
            "eta": 0.05,
            "max_depth": 10,
            "tree_method": "gpu_hist",
        }
        # GPU accelerated training
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dtest = xgb.DMatrix(X_valid, label=y_valid, feature_names=feature_names)
        logger.info('Data loaded! Model training starts...')
        model = xgb.train(param, dtrain,num_round)
        logger.info('Model trained! Shap Values Prediction starts...')
        model.set_param({"predictor": "gpu_predictor"})
        explainer_train = shap.TreeExplainer(model)
        shap_values_train = explainer_train.shap_values(X_train)
        explainer_test = shap.TreeExplainer(model)
        shap_values_test = explainer_test.shap_values(X_valid)
        logger.info('Training Completed!!!')
        other_info = {}
        dataset = {}
        dataset = {'X_train': X_train, 'X_valid': X_valid, 'y_train': y_train, 'y_valid': y_valid}
        shap_values = {'shap_values_train': shap_values_train, 'shap_values_test': shap_values_test, }
        y_pred_test = model.predict(dtest)
        y_pred_train = model.predict(dtrain)
        print ('Auc Score Test, Train:', roc_auc_score(y_valid, y_pred_test), roc_auc_score(y_train, y_pred_train))
        if replication_set is not None:
            
            y_rep_pred = model.predict(drep)
            dataset['X_rep'] = replication_dataframe
            dataset['y_rep'] = y_rep_true
            other_info['y_rep'] = y_rep_true
            other_info['y_pred_rep'] = y_rep_pred
            other_info['ID_rep'] = replication_set['ID'] 
            other_info['AUC_rep'] = roc_auc_score( y_rep_true, y_rep_pred )
            print ('='*20, "REP")
            print ( roc_auc_score( y_rep_true, y_rep_pred) )
            explainer_rep = shap.TreeExplainer(model)
            shap_values_rep = explainer_rep.shap_values(replication_dataframe)
            shap_values['shap_values_rep'] = shap_values_rep
        other_info['ID_train'] = ID_train
        other_info['ID_test'] = ID_test
        other_info['y_pred_test'] = y_pred_test
        other_info['y_pred_train'] = y_pred_train
        other_info['y_test'] = y_valid
        other_info['y_train'] = y_train
        other_info['AUC_train'] = roc_auc_score(y_train, y_pred_train)
        other_info['AUC_test'] = roc_auc_score(y_valid, y_pred_test)
        expected_values = {'explainer_train': explainer_train.expected_value,  'explainer_test': explainer_test.expected_value}
        return model, (shap_values, dataset, expected_values, other_info)

    def trainXGBModel_multiclass(self, data, feature_names, label_name, replication_set=None):
        logger.info('Training starts...')
        X_train, X_valid, y_train, y_valid = train_test_split(data, data[label_name] , test_size=0.05, random_state=42)
        ID_train = X_train['ID']
        ID_test = X_valid['ID']
        X_train = X_train[feature_names].copy()
        X_valid = X_valid[feature_names].copy()
        num_round = 500
        param = {
            'objective':'multi:softprob',
            "eta": 0.05,
            "max_depth": 10,
            "tree_method": "gpu_hist",
            "num_class": 6,
            "disable_default_eval_metric": 1
        }
        # GPU accelerated training
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dtest = xgb.DMatrix(X_valid, label=y_valid, feature_names=feature_names)
        logger.info('Data loaded! Model training starts...')
        model = xgb.train(param, dtrain,num_round)
        logger.info('Model trained! Shap Values Prediction starts...')
        model.set_param({"predictor": "gpu_predictor"})
        explainer_train = shap.TreeExplainer(model)
        shap_values_train = explainer_train.shap_values(X_train)
        explainer_test = shap.TreeExplainer(model)
        shap_values_test = explainer_test.shap_values(X_valid)
        logger.info('Training Completed!!!')
        other_info = {}
        dataset = {}
        dataset = {'X_train': X_train, 'X_valid': X_valid, 'y_train': y_train, 'y_valid': y_valid}
        shap_values = {'shap_values_train': shap_values_train, 'shap_values_test': shap_values_test, }
        y_pred_test = model.predict(dtest)
        y_pred_train = model.predict(dtrain)
        print ('='*20, "Train")
        print (calculate_roc_auc_score_multiclass( y_pred_train, y_train))
        print ('='*20, "Test")
        print (calculate_roc_auc_score_multiclass( y_pred_test, y_valid))
        if replication_set is not None:
            replication_dataframe = replication_set[feature_names].copy()
            y_rep_true = replication_set[label_name] 
            drep = xgb.DMatrix(replication_dataframe, label=y_rep_true, feature_names=feature_names)
            y_rep_pred = model.predict(drep)
            dataset['X_rep'] = replication_dataframe
            dataset['y_rep'] = y_rep_true
            other_info['y_rep'] = y_rep_true
            other_info['y_pred_rep'] = y_rep_pred
            other_info['ID_rep'] = replication_set['ID'] 
            other_info['AUC_rep'] = calculate_roc_auc_score_multiclass( y_rep_pred, y_rep_true )
            print ('='*20, "REP")
            print ( calculate_roc_auc_score_multiclass( y_rep_pred, y_rep_true ) )
            explainer_rep = shap.TreeExplainer(model)
            shap_values_rep = explainer_rep.shap_values(replication_dataframe)
            shap_values['shap_values_rep'] = shap_values_rep
        print ('='*20, "Stop")
        other_info['ID_train'] = ID_train
        other_info['ID_test'] = ID_test
        other_info['y_pred_test'] = y_pred_test
        other_info['y_pred_train'] = y_pred_train
        other_info['y_test'] = y_valid
        other_info['y_train'] = y_train
        other_info['AUC_train'] = calculate_roc_auc_score_multiclass( y_pred_train, y_train) 
        other_info['AUC_test'] = calculate_roc_auc_score_multiclass( y_pred_test, y_valid) 
        expected_values = {'explainer_train': explainer_train.expected_value,  'explainer_test': explainer_test.expected_value}
        return shap_values, dataset, expected_values, other_info
      
    def trainLightGBMModel_multiclass(self, data, feature_names, label_name, replication_set=None):
        logger.info('Training starts...')
        X_train, X_valid, y_train, y_valid = train_test_split(data, data[label_name] , test_size=0.3, random_state=42)
        ID_train = X_train['ID']
        ID_test = X_valid['ID']
        X_train = X_train[feature_names].copy()
        X_valid = X_valid[feature_names].copy()
        num_round = 100
        params={}
        params['learning_rate']=0.03
        params['boosting_type']='gbdt' #GradientBoostingDecisionTree
        params['objective']='multiclass' #Multi-class target feature
        params['metric']='multi_logloss' #metric for multi-class
        params['max_depth']=10
        params['num_class']=6 #no.of unique values in the target class not inclusive of the end value
        # GPU accelerated training
        dataset = {'X_train': X_train, 'X_valid': X_valid, 'y_train': y_train, 'y_valid': y_valid}
        other_info = {}
        dtrain = lgb.Dataset(X_train, label=y_train)
        dtest = lgb.Dataset(X_valid, label=y_valid)
        logger.info('Data loaded! Model training starts...')
        model = lgb.train(params, dtrain,num_round)
        logger.info('Model trained! Shap Values Prediction starts...')
        explainer_train = shap.TreeExplainer(model)
        shap_values_train = explainer_train.shap_values(X_train)
        explainer_test = shap.TreeExplainer(model)
        shap_values_test = explainer_test.shap_values(X_valid)
        logger.info('Training Completed!!!')
        shap_values = {'shap_values_train': shap_values_train, 'shap_values_test': shap_values_test, }
        y_pred_test = model.predict(X_valid)
        y_pred_train = model.predict(X_train)
        print ('='*20, "Train")
        print (calculate_roc_auc_score_multiclass( y_pred_train, y_train))
        print ('='*20, "Test")
        print (calculate_roc_auc_score_multiclass( y_pred_test, y_valid))
        if replication_set is not None:
            replication_dataframe = replication_set[feature_names].copy()
            y_rep_true = replication_set[label_name] 
            y_rep_pred = model.predict(replication_dataframe)
            dataset['X_rep'] = replication_dataframe
            dataset['y_rep'] = y_rep_true
            other_info['y_rep'] = y_rep_true
            other_info['y_pred_rep'] = y_rep_pred
            other_info['ID_rep'] = replication_set['ID'] 
            other_info['AUC_rep'] = calculate_roc_auc_score_multiclass( y_rep_pred, y_rep_true )
            print ('='*20, "REP")
            print ( calculate_roc_auc_score_multiclass( y_rep_pred, y_rep_true ) )
            explainer_rep = shap.TreeExplainer(model)
            shap_values_rep = explainer_rep.shap_values(replication_dataframe)
            shap_values['shap_values_rep'] = shap_values_rep
        print ('='*20, "Stop")
        other_info['ID_train'] = ID_train
        other_info['ID_test'] = ID_test
        other_info['y_pred_test'] = y_pred_test
        other_info['y_pred_train'] = y_pred_train
        other_info['y_test'] = y_valid
        other_info['y_train'] = y_train
        other_info['AUC_train'] = calculate_roc_auc_score_multiclass( y_pred_train, y_train) 
        other_info['AUC_test'] = calculate_roc_auc_score_multiclass( y_pred_test, y_valid) 
        expected_values = {'explainer_train': explainer_train.expected_value,  'explainer_test': explainer_test.expected_value}
        return shap_values, dataset, expected_values, other_info