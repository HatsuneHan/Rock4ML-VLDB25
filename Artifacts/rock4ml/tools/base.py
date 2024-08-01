import json
import os
import pandas as pd
import IPython.display
import raha
import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

class Cleaning:
  def __init__(self, root_dir, dataset, label_column, verbose = False, labeling_budget = 20, iter_cnt = None, multi_discovery = False, rock_port = None):
    # dataset path
    self.ROOT_DIR = root_dir
    self.ROOT_TMP_DIR = "/tmp/"
    self.DATASET = dataset
    self.LABEL_COLUMN = label_column
    self.MULTI_DISCOVERY = multi_discovery

    # use for raha and baran
    self.VERBOSE = verbose
    self.LABELING_BUDGET = labeling_budget
    self.SAVE_DIR = os.path.join(root_dir, dataset)
    self.ITER = iter_cnt
    self.ROCK_PORT = rock_port

    if iter_cnt is None:
      self.DATASET_DICT = {
        "name": dataset,
        "path": root_dir + '/' + dataset + "/repaired/" + dataset + "_dirty.csv", # dirty_path
        "clean_path": root_dir + '/' + dataset + "/repaired/" + dataset + "_clean.csv" # clean_path
      }
    else:
      self.DATASET_DICT = {
        "name": dataset,
        "path": root_dir + '/' + dataset + "/repaired/" + dataset + "_dirty_{}.csv".format(iter_cnt), # dirty_path
        "clean_path": root_dir + '/' + dataset + "/repaired/" + dataset + "_clean.csv" # clean_path
      }

  def mixRahaBaran(self):  
    
    print("Dataset Path:")
    print(self.DATASET_DICT)

    # init raha and baran
    myraha = raha.Detection()
    mybaran = raha.Correction()

    # set the parameters for raha and baran
    myraha.LABELING_BUDGET = self.LABELING_BUDGET
    myraha.SAVE_RESULTS = False
    mybaran.LABELING_BUDGET = self.LABELING_BUDGET
    myraha.VERBOSE = self.VERBOSE
    mybaran.VERBOSE = self.VERBOSE
    myraha.SAVE_RESULTS = False

    # initialize the dataset for raha
    d = myraha.initialize_dataset(self.DATASET_DICT)
    print(d.dataframe.head())
    # the beginning of raha
    # step 1: run each strategy
    myraha.run_strategies(d)

    # step 2: generate features according to the results from different strategies
    myraha.generate_features(d)

    # step 3: clust the feature vectors
    myraha.build_clusters(d)
    
    while len(d.labeled_tuples) < myraha.LABELING_BUDGET:
      # step 4: sample from each cluster 
      myraha.sample_tuple(d)

      # step 5a: label each representative (use existing ground truth)
      if d.has_ground_truth:
        myraha.label_with_ground_truth(d)
      # step 5b: label each representative (otherwise, display a gui to label manually)
      else:
        print("Label the dirty cells in the following sampled tuple.")
        sampled_tuple = pd.DataFrame(data=[d.dataframe.iloc[d.sampled_tuple, :]], columns=d.dataframe.columns)
        IPython.display.display(sampled_tuple)
        for j in range(d.dataframe.shape[1]):
          cell = (d.sampled_tuple, j)
          value = d.dataframe.iloc[cell]
          correction = input("What is the correction for value '{}'? Type in the same value if it is not erronous.\n".format(value))
          user_label = 1 if value != correction else 0
          d.labeled_cells[cell] = [user_label, correction]
        d.labeled_tuples[d.sampled_tuple] = 1

    # step 6: propagate manual labels in the clusters
    myraha.propagate_labels(d)

    # step 7: train and predict the rest of data cells
    myraha.predict_labels(d)
    # the end of raha

    # initialize the dataset for baran
    d = mybaran.initialize_dataset(d)

    # initialize the models for baran
    mybaran.initialize_models(d)

    # step 1: set labeled tuples of baran to be the same as raha
    for si in d.labeled_tuples:
      # step 2: iteratively update the models
      d.sampled_tuple = si
      mybaran.update_models(d)
      mybaran.predict_corrections(d)

    repair_df = d.dataframe.copy()
    for cell in d.corrected_cells:
      repair_df.iloc[cell] = d.corrected_cells[cell]

    p, r, f = d.get_data_cleaning_evaluation(d.corrected_cells)[-3:]
    print("Baran's performance on {}:\nPrecision = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}".format(d.name, p, r, f))  
    
    return repair_df
  
  def rock(self):
    if self.ITER != 0:
      dirty_path = self.ROOT_TMP_DIR + "/" + self.DATASET + "/repaired/" + self.DATASET + "_dirty_" + str(self.ITER) + ".csv"
    else:
      dirty_path = self.ROOT_TMP_DIR + "/" + self.DATASET + "/repaired/" + self.DATASET + "_dirty.csv"

    if self.MULTI_DISCOVERY == False:
      if self.ITER == 0:
        save_rule_path = self.ROOT_TMP_DIR + "/" + self.DATASET + "/repaired/" + self.DATASET + "_rule.csv"
        rule_path = None
      else:
        rule_path = self.ROOT_TMP_DIR + "/" + self.DATASET + "/repaired/" + self.DATASET + "_rule.csv"
        save_rule_path = None
    else:
      save_rule_path = self.ROOT_TMP_DIR + "/" + self.DATASET + "/repaired/" + self.DATASET + "_rule_" + str(self.ITER) + ".csv"
      rule_path = None

    clean_path = self.ROOT_TMP_DIR + "/" + self.DATASET + "/repaired/" + self.DATASET + "_clean.csv"

    if self.ITER == 0:
      output_path = self.ROOT_TMP_DIR + "/" + self.DATASET + "/repaired/" + self.DATASET + "_repaired_rock.csv"
    else:
      output_path = self.ROOT_TMP_DIR + "/" + self.DATASET + "/repaired/" + self.DATASET + "_repaired_rock_" + str(self.ITER) + ".csv"

    if self.DATASET == "adult":
      data = {
        "taskId": 110042 + self.ITER,
        "csvInfo": {
            "tableName": "adult",
            "path": dirty_path,
            "cleanPath": clean_path,
            "columnType": {
                "age": "float64",
                "workclass": "string",
                "fnlwgt": "string",
                "education": "string",
                "educational-num": "float64",
                "marital-status": "string",
                "occupation": "string",
                "relationship": "string",
                "race": "string",
                "gender": "string",
                "capital-gain": "float64",
                "capital-loss": "float64",
                "hours-per-week": "float64",
                "native-country": "string",
                "income": "string"
            }
        },
        "rds": {
            "taskID": 110042 + self.ITER,
            "tablesID": [],
            "conf": {
                "EnumSize":45,
                "TopKLayer":10,
                "TreeLevel":4,
                "DecisionTreeMaxDepth":10,
                "Confidence":0.8,
                "Support":0.001
            },
            "SkipYColumns": ["income", "fnlwgt"]
        },
        "correctInfo":{
            "maxCorrectInChase":1200,
            "maxChase": 120,
            "minCorrectRatio":0.5,
            "topRuleNumber":50
        },
        "resultSavePath": output_path,
        "ruleSavePath": save_rule_path, 
        "rulePath": rule_path
      }
    elif self.DATASET == "german":
      data = {
        "taskId": 140015 + self.ITER,
        "csvInfo": {
            "tableName": "german",
            "path": dirty_path,
            "cleanPath": clean_path,
            "columnType": {
                "Status of existing checking account": "string",
                "Duration": "float64",
                "Credit history": "string",
                "Purpose": "string",
                "Credit amount": "float64",
                "Savings account/bonds": "string",
                "Present employment since": "string",
                "Installment rate": "float64",
                "personal status": "string",
                "Other debtors": "string",
                "Present residence since": "float64",
                "Property": "string",
                "Age": "float64",
                "Other installment plans": "string",
                "Housing": "string",
                "Number of existing credits": "float64",
                "Job": "string",
                "Number of people": "float64",
                "Telephone": "string",
                "foreign worker": "string",
                "class": "string"
            }
        },
        "rds": {
            "taskID": 140015 + self.ITER,
            "tablesID": [],
            "conf": {
                "EnumSize":45,
                "TopKLayer":10,
                "TreeLevel":3,
                "DecisionTreeMaxDepth":10,
                "Confidence":0.85,
                "Support":0.001
            },
            "SkipYColumns": ["class"]
        },
        "correctInfo":{
            "maxCorrectInChase":55,
            "maxChase": 120,
            "minCorrectRatio":0.5,
            "topRuleNumber":200
        },
        "resultSavePath": output_path,
        "ruleSavePath": save_rule_path, 
        "rulePath": rule_path
      }
    elif self.DATASET == "default":
      data = {
        "taskId": 130247 + self.ITER,
        "csvInfo": {
            "tableName": "default",
            "path": dirty_path,
            "cleanPath": clean_path,
            "columnType": {
                "LIMIT_BAL": "float64",
                "SEX": "string",
                "EDUCATION": "string",
                "MARRIAGE": "string",
                "AGE": "float64",
                "PAY_0": "string",
                "PAY_2": "string",
                "PAY_3": "string",
                "PAY_4": "string",
                "PAY_5": "string",
                "PAY_6": "string",
                "BILL_AMT1": "float64",
                "BILL_AMT2": "float64",
                "BILL_AMT3": "float64",
                "BILL_AMT4": "float64",
                "BILL_AMT5": "float64",
                "BILL_AMT6": "float64",
                "PAY_AMT1": "float64",
                "PAY_AMT2": "float64",
                "PAY_AMT3": "float64",
                "PAY_AMT4": "float64",
                "PAY_AMT5": "float64",
                "PAY_AMT6": "float64",
                "default.payment.next.month": "string"
            }
        },
        "rds": {
            "taskID": 130247 + self.ITER,
            "tablesID": [],
            "conf": {
                "EnumSize":45,
                "TopKLayer":10,
                "TreeLevel":4,
                "DecisionTreeMaxDepth":10,
                "Confidence":0.85,
                "Support":0.01
            },
            "SkipYColumns": ["default.payment.next.month"]
        },
        "correctInfo":{
            "maxCorrectInChase":2000,
            "maxChase": 100,
            "minCorrectRatio":0.5,
            "topRuleNumber":60
        },
        "resultSavePath": output_path,
        "ruleSavePath": save_rule_path, 
        "rulePath": rule_path
      }
    elif self.DATASET == "nursery":
      data = {
        "taskId": 180047 + self.ITER,
        "csvInfo": {
            "tableName": "nursery",
            "path": dirty_path,
            "cleanPath": clean_path,
            "columnType": {
                "parents": "string",
                "has_nurs": "string",
                "form": "string",
                "children": "string",
                "housing": "string",
                "finance": "string",
                "social": "string",
                "health": "string",
                "final evaluation": "string"
            }
        },
        "rds": {
            "taskID": 180047 + self.ITER,
            "tablesID": [],
            "conf": {
                "DecisionTreeMaxRowSize": 1000,
                "EnumSize": 45,
                "TopKLayer": 10,
                "TreeLevel": 4,
                "Confidence": 0.9,
                "Support": 0.01,
                "DecisionTreeMaxDepth": 10
            },
            "SkipYColumns": ["final evaluation"]
        },
        "correctInfo":{
            "maxChase": 80,
            "minCorrectRatio": 0.5,
            "maxCorrectInChase": 200,
            "topRuleNumber": 10
        },
        "enableEnum": True,
        "resultSavePath": output_path,
        "ruleSavePath": save_rule_path, 
        "rulePath": rule_path
      }
    elif self.DATASET == "Bank":
      data = {
        "taskId": 138247 + self.ITER,
        "csvInfo": {
            "tableName": "Bank",
            "path": dirty_path,
            "cleanPath": clean_path,
            "columnType": {
                "age": "float64",
                "job": "string",
                "marital": "string",
                "education": "string",
                "default": "string",
                "balance": "float64",
                "housing": "string",
                "loan": "string",
                "contact": "string",
                "day": "float64",
                "month": "string",
                "duration": "float64",
                "campaign": "float64",
                "pdays": "float64",
                "previous": "float64",
                "poutcome": "string",
                "y":"string"
            }
        },
        "rds": {
            "taskID": 138247 + self.ITER,
            "tablesID": [],
            "conf": {
                "EnumSize": 45,
                "TopKLayer": 10,
                "TreeLevel": 3,
                "Confidence": 0.95,
                "Support": 0.01,
                "DecisionTreeMaxDepth": 10
            },
            "SkipYColumns": ["y", "poutcome"]
        },
        "correctInfo":{
            "maxChase": 120,
            "minCorrectRatio": 0.5,
            "maxCorrectInChase": 1500,
            "topRuleNumber": 150
        },
        "resultSavePath": output_path,
        "ruleSavePath": save_rule_path, 
        "rulePath": rule_path
      }

    elif self.DATASET == "road_safety":
      data = {
        "taskId": 190240 + self.ITER,
        "csvInfo": {
            "tableName": "roadsafety",
            "path": dirty_path,
            "cleanPath": clean_path,
            "columnType": {
                "Vehicle_Reference_df_res": "float64",
                "Vehicle_Type": "string",
                "Vehicle_Manoeuvre": "string",
                "Vehicle_Location-Restricted_Lane": "string",
                "Hit_Object_in_Carriageway": "string",
                "Hit_Object_off_Carriageway": "string",
                "Was_Vehicle_Left_Hand_Drive": "string",
                "Age_of_Driver": "float64",
                "Age_Band_of_Driver": "string",
                "Engine_Capacity": "float64",
                "Propulsion_Code": "string",
                "Age_of_Vehicle": "float64",
                "Location_Easting_OSGR": "float64",
                "Location_Northing_OSGR": "float64",
                "Longitude": "float64",
                "Latitude": "float64",
                "Police_Force": "float64",
                "Number_of_Vehicles": "string",
                "Number_of_Casualties": "string",
                "Local_Authority": "float64",
                "1st_Road_Number": "float64",
                "2nd_Road_Number": "float64",
                "Urban_or_Rural_Area": "string",
                "Vehicle_Reference_df": "float64",
                "Casualty_Reference": "float64",
                "Sex_of_Casualty": "string",
                "Age_of_Casualty": "float64",
                "Age_Band_of_Casualty": "string",
                "Pedestrian_Location": "string",
                "Pedestrian_Movement": "string",
                "Casualty_Type": "float64",
                "Casualty_IMD_Decile": "string",
                "SexofDriver": "string"
            }
        },
        "rds": {
            "taskID": 190240 + self.ITER,
            "tablesID": [],
            "conf": {
                "EnumSize": 45,
                "TopKLayer": 10,
                "TreeLevel": 3,
                "Confidence": 0.6,
                "Support": 0.0001,
                "DecisionTreeMaxDepth": 5
            },
            "skipYColumns": [
                "SexofDriver"
            ]
        },
        "correctInfo": {
            "maxChase": 100,
            "minCorrectRatio": 0.5,
            "maxCorrectInChase": 4000,
            "topRuleNumber": 10
        },

        "resultSavePath": output_path,
        "ruleSavePath": save_rule_path, 
        "rulePath": rule_path
      }
    
    else:
      raise ValueError("Dataset not supported")
    
    data = json.dumps(data)
    
    ip_addr = "localhost:"
    response = requests.post(ip_addr + str(self.ROCK_PORT) + "/rock-4-ml", data=data)

    print(response.text)