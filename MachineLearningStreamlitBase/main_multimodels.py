from MachineLearningStreamlitBase.train_model import generateFullData
import pandas as pd
import numpy as np
import os
import pickle

disease_list = ['Dementia', 'Alz', 'Vascular', 'Parkinsonism', 'PD']

for disease in disease_list:
    print ('='*20, disease, '='*20)

    ppmi_traineddata_path = "drug_mine_2020/{}_UKB_Janaury2021.dataForML.h5".format(disease)
    column_name_path = "drug_mine_2020/{}_UKB_Janaury2021.list_features.txt".format(disease)
    with open(column_name_path, 'r') as the_file:
        st = the_file.read().strip().split('\n')
        d = {i:1 for i in st}
        k = [i for i in st if i not in ['ID', 'PHENO']]
    
    ppmi_traineddata_df = pd.read_hdf(ppmi_traineddata_path, key='dataForML', mode='r')
    matchingCols_file = open(column_name_path, "r")
    matching_column_names_list = matchingCols_file.read().splitlines()
    ppmi_traineddata_df = ppmi_traineddata_df[np.intersect1d(ppmi_traineddata_df.columns, matching_column_names_list)]
    obj = generateFullData()
    model, train = obj.trainXGBModel_binaryclass(ppmi_traineddata_df, k, 'PHENO')
    os.makedirs('saved_models', exist_ok=True)
    with open('saved_models/trainXGB_gpu_{}.model'.format(disease), 'wb') as f:
        pickle.dump(model, f)
    with open('saved_models/trainXGB_gpu_{}.data'.format(disease), 'wb') as f:
        pickle.dump(train, f)

test_ids = {}
result_aucs = {}
for disease in disease_list:
    with open('saved_models/trainXGB_gpu_{}.data'.format(disease), 'rb') as f:
        temp = pickle.load(f)
    result_aucs[disease] = (temp[3]['AUC_train'], temp[3]['AUC_test'])
    print (disease, result_aucs[disease])

with open('saved_models/trainXGB_gpu.aucs', 'wb') as f:
        pickle.dump(result_aucs, f)

col_dict_map = {}
with open('saved_models/trainXGB_categorical_map.pkl', 'wb') as f:
    pickle.dump(col_dict_map, f)

with open('saved_models/trainXGB_class_map.pkl', 'wb') as f:
    pickle.dump(disease_list, f)
