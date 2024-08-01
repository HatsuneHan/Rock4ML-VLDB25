import copy
import math
import os
import random
import time
import lightgbm as lgb
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
import tensorflow as tf
from tools.preprocess import Preprocessor
from tools.base import Cleaning
from tools.ga import *
from tools.coreset import *
import gc
from deel.influenciae.common import InfluenceModel, ExactIHVP, ConjugateGradientDescentIHVP
from tensorflow.keras.losses import CategoricalCrossentropy, Reduction
from deel.influenciae.influence import FirstOrderInfluenceCalculator
from deel.influenciae.utils import ORDER
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import f1_score,precision_score,recall_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import clone_model
from tensorflow.keras.utils import CustomObjectScope

from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from tools.tabtransformer import *
from tools.transformer.utils.preprocessing import df_to_dataset_np_cat_label
from tools.transformer.models.fttransformer import FTTransformerEncoder, FTTransformer

class Rock4ML:

  def __init__(self, dataset_params: dict, model_params: dict):

    self.model_name = model_params['model_name']
    self.dataset_name = dataset_params['name']
    self.dataset_rootdir = dataset_params['dataset_rootdir']

    self.label_column = dataset_params['label_column']
    self.cr_method = model_params['cr_method']
    
    self.maxiter_ga = model_params['maxiter_ga']

    self.no_epoch = model_params['no_epoch']

    if model_params['inc_epoch'] == 0:
      self.inc_epoch = max(int(self.no_epoch / 10), 1)
    else:
      self.inc_epoch = model_params['inc_epoch']

    self.random_state = model_params['random_state']
    self.batch_size = model_params['batch_size']
    self.temperature = model_params['temperature']

    self.kinf_ratio = model_params['kinf_ratio']
    self.rock_port = model_params['rock_port']
    self.max_epoch_rock4ml = 15

    # this is the ratio for m, which means m = topk_ga_ratio * |schema|
    self.topk_ga_ratio = model_params['topk_ga']

  def fit(self,
          X_train: pd.DataFrame,
          y_train: pd.DataFrame,  
          X_test: pd.DataFrame,
          y_test: pd.DataFrame,
          X_train_clean: pd.DataFrame):

    self.acc_dict = {}
    
    np.random.seed(self.random_state)
    random.seed(self.random_state)
    tf.random.set_seed(self.random_state)
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
      except RuntimeError as e:
          print(e)

    print("Incremental Epoch is", self.inc_epoch)
    # prepare dataset
    self.X_train = X_train
    self.y_train = y_train
    self.X_train_clean = X_train_clean

    if self.topk_ga_ratio != 0:
      self.topk_ga = int(self.topk_ga_ratio * len(self.X_train.columns))
      self.maxoffspring_ga = int(self.topk_ga * (self.topk_ga-1) / 2)
    else:
      # default m is |schema|
      self.topk_ga = len(self.X_train.columns)
      self.maxoffspring_ga = int(self.topk_ga * (self.topk_ga-1) / 2)
    
    self.topk_inf = math.ceil(self.kinf_ratio * len(self.X_train))
    
    self.clean_flag = pd.DataFrame(False, index=self.X_train.index, columns=self.X_train.columns)
    self.attrnum = 0

    self.X_test = X_test
    self.y_test = y_test

    # multi-class -> multi_logloss
    # binary-class -> binary_logloss
    self.metric = self.getMetric(y_train)

    self.categorical_features, self.numerical_features, self.all_features = self.getFeatures()

    self.endflag = 0
    self.substart = 0

    # get preprocessor using X_train and y_train
    if self.model_name != "FTTransformer":
      self.preprocessor = self.getPreprocessor(self.X_train, self.y_train)
      self.X_test_embed, self.y_test_embed = self.transformData(self.preprocessor, self.X_test, self.y_test)
      
      self.categorical_column_dicts = self.outputshape = None
      self.X_embed, self.y_embed = self.transformData(self.preprocessor, self.X_train, self.y_train)
      self.train_ds = getTrainDS(self.X_embed, self.y_embed, self.batch_size)
      self.transformerpreprocessor = None

    else:
      self.preprocessor = None
      
      # currently this FTTransformer model cannot process unseen data
      # so when we get the preprocessor, we need to use the whole data instead of X_train
      # so that for each element in the unseen data, we can get the corresponding embedding
      # For other models, we can use X_train to get the preprocessor

      self.transformerpreprocessor = self.getTransformerPreprocessor(pd.concat([self.X_train, self.X_test], axis=0), pd.concat([self.y_train, self.y_test], axis=0))
      self.X_embed = self.y_embed = None
      self.train_ds, self.outputshape, self.categorical_column_dicts, _, self.y_embed = getTransformerDS(self.X_train, self.y_train, self.transformerpreprocessor, self.numerical_features, self.categorical_features, self.label_column, self.batch_size)
      
      self.test_ds, _, _, _, self.y_test_embed = getTransformerDS(self.X_test, self.y_test, self.transformerpreprocessor, self.numerical_features, self.categorical_features, self.label_column, self.batch_size)

    iter_cnt = 0

    self.optimizer = Adam(
      learning_rate=1e-3
    )

    pre_losses = None
    self.idxClean = pd.Index([])

    while self.endflag != 1:
      print("-" * 60)
      print(f"iter_cnt: {iter_cnt}")

      # fit model
      if iter_cnt == 0:
        self.model, self.sub_model, self.optimizer = trainModel(self.train_ds, self.X_embed, self.y_embed, self.model_name, self.optimizer, self.no_epoch, self.random_state, 
                                                                self.numerical_features, self.categorical_features, self.categorical_column_dicts, self.X_train, self.outputshape) # init the model
      else:
        # unlearning
        last_old_weights = self.model_weights[-2:]
        epsilon = 1 / len(self.X_train)
        last_new_weights = [last_old_weights[i] + epsilon * self.unlearning_vector[i] for i in range(len(last_old_weights))]

        unlearning_weights = copy.deepcopy(self.model_weights)
        unlearning_weights[-2:] = last_new_weights

        self.model.set_weights(unlearning_weights)

        # incrementally finetune
        self.model.fit(self.train_ds, epochs=self.inc_epoch, verbose=0, shuffle = False)   

      if self.model_name == "FTTransformer":
        self.losses = tf.keras.losses.categorical_crossentropy(self.y_embed, self.model(self.train_ds), from_logits=False).numpy()
      else:
        self.losses = tf.keras.losses.categorical_crossentropy(self.y_embed, self.model(self.X_embed, training=False).numpy(), from_logits=False).numpy()

      if iter_cnt == 0:
        pre_losses = None
        self.acuml_losses = self.losses
        self.idxClean = pd.Index([])
      else:
        pre_losses = copy.deepcopy(self.acuml_losses)
        self.acuml_losses = dynamic_loss(self.acuml_losses, self.losses, 0.5)
        self.idxClean = self.identifyCleanData(pre_losses, iter_cnt)  

        if len(list(self.idxClean)) > 0:
          if self.model_name == "FTTransformer":
            self.train_clean_ds, _, _, _, _ = getTransformerDS(self.X_train.loc[list(self.idxClean)], self.y_train.loc[list(self.idxClean)], self.transformerpreprocessor, self.numerical_features, self.categorical_features, self.label_column, self.batch_size)
          else:
            X_clean_embed, y_clean_embed = self.X_embed[list(self.idxClean)], self.y_embed[list(self.idxClean)]
            self.train_clean_ds = getTrainDS(X_clean_embed, y_clean_embed, self.batch_size)

        # self.sub_model.fit(self.train_clean_ds, epochs=self.inc_epoch, verbose=0, shuffle = False)
        # self.substart = 1
      
      self.model_weights = self.model.get_weights()

      if self.model_name == "FTTransformer":
        self.init_scores = self.model.predict(self.train_ds, verbose=0)
        self.test_scores = self.model.predict(self.test_ds, verbose=0)
      else:
        self.init_scores = predictProb(self.X_embed, self.model)
        self.test_scores = predictProb(self.X_test_embed, self.model)

      eval_score = evalScore(self.test_scores, self.y_test_embed)['F1']
      
      if iter_cnt == 0:
        self.acc_dict['initial_dirty'] = eval_score
        self.dirty_score = eval_score
      else:
        self.acc_dict['iter_' + str(iter_cnt)] = eval_score

      # preprocess for lightgbm in getCriticalAttr
      # for binary, init_scores use shape = n_sample * 1 (here not n_class)
      # for multi, init_scores use shape = n_sample * n_class 

      if isinstance(self.init_scores, np.ndarray):
        self.init_scores = pd.DataFrame(self.init_scores)

      label_num = len(set(self.y_train.values.flatten()))

      if label_num == 2:
        self.init_scores = self.init_scores.iloc[:, :label_num-1]
      else:
        self.init_scores = self.init_scores
      
      ##### get coreset
      self.idxT, self.idxV = self.getCoreset(self.temperature, self.random_state)
      
      ##### CRMethod Repair
      start_time_cr = time.time()
      self.repair_dataframe = self.cleanDataWithCR(iter_cnt, self.rock_port, False)
      self.repair_dataframe.replace('<nil>', np.nan, inplace=True)

      # print("CRMethod Repair Time: ", time.time() - start_time_cr)

      # get the repair data
      self.X_train_repair = self.repair_dataframe.drop(self.label_column, axis=1)

      # modify to get actual X_train_repair according to the clean_flag since a data cell can only be repaired once

      self.X_train_repair = self.X_train_repair.where(~self.clean_flag, self.X_train)
      self.y_train_repair = pd.DataFrame(self.repair_dataframe[self.label_column])
      
      # use the repair data itself to get the preprocessor to get its f1 in the first iter
      # this f1 is the f1 of the cr_method, you can report it directly
      if iter_cnt == 0:
        # to ensure fairness, we need to use a new preprocessor for the cr_method repair data
        if self.model_name == "FTTransformer":
          cr_preprocessor = self.getTransformerPreprocessor(pd.concat([self.X_train_repair, self.X_test], axis = 0), 
                                                            pd.concat([self.y_train_repair, self.y_test], axis = 0))
          cr_optimizer = Adam(learning_rate=1e-3)
          train_ds_repair, cr_outputshape, cr_categorical_column_dicts, _, y_repair_embed = getTransformerDS(self.X_train_repair, self.y_train_repair, cr_preprocessor, self.numerical_features, self.categorical_features, self.label_column, self.batch_size)
          
          X_repair_embed = y_repair_embed = None
          cr_model, _, cr_optimizer = trainModel(train_ds_repair, X_repair_embed, y_repair_embed, self.model_name, cr_optimizer, self.no_epoch, self.random_state, 
                                                 self.numerical_features, self.categorical_features, cr_categorical_column_dicts,
                                                 self.X_train_repair, cr_outputshape) # init the model

          cr_test_ds, _, _, _, cr_y_test_embed = getTransformerDS(self.X_test, self.y_test, cr_preprocessor, self.numerical_features, self.categorical_features, self.label_column, self.batch_size)
          cr_test_scores = cr_model.predict(cr_test_ds, verbose=0)

          cr_eval_score = evalScore(cr_test_scores, cr_y_test_embed)['F1']
          
        else:
          cr_preprocessor = self.getPreprocessor(self.X_train_repair, self.y_train_repair)
          cr_optimizer = Adam(learning_rate=1e-3)
          X_repair_embed, y_repair_embed = self.transformData(cr_preprocessor, self.X_train_repair, self.y_train_repair)
          test_X_embed_cr, test_y_embed_cr = self.transformData(cr_preprocessor, self.X_test, self.y_test)
          train_ds_repair = getTrainDS(X_repair_embed, y_repair_embed, self.batch_size)

          cr_model, _, cr_optimizer = trainModel(train_ds_repair, X_repair_embed, y_repair_embed, self.model_name, cr_optimizer, self.no_epoch, self.random_state) # init the model
          cr_test_scores = predictProb(test_X_embed_cr, cr_model)
          cr_eval_score = evalScore(cr_test_scores, test_y_embed_cr)['F1']
          
        self.cr_score = cr_eval_score
        self.acc_dict['cr_method'] = cr_eval_score
            
      # here we use the preprocessor from the raw data to continue (influential tuple and critical attr)
      # since we only train the model one time at the start, i think we should keep the preprocessor the same from start to end
      if self.model_name != "FTTransformer":
        self.X_repair_embed, self.y_repair_embed = self.transformData(self.preprocessor, self.X_train_repair, self.y_train_repair)
      else:
        self.X_repair_embed = self.y_repair_embed = None
      

      ##### get influential tuples:    
      influ_tuples_dict = self.getInfluentialTuples(iter_cnt)
      influ_tuples_dict = dict(list(influ_tuples_dict.items())[:self.topk_inf]) # get the top-k tuples
      print(f"Selected Influential Tuples Length: {len(influ_tuples_dict)}")

      #### get critical attrs
      criti_attr_dict, score_raw = self.getCriticalAttrs()
      criti_attr_dict = dict(list(criti_attr_dict.items())[:1]) # get the top-1 combs
      if(criti_attr_dict[list(criti_attr_dict.keys())[0]] < 0 or criti_attr_dict[list(criti_attr_dict.keys())[0]] <= score_raw + 1e-4): # make sure benefit from cleaning
        criti_attr_dict = {}

      # criti_attr_dict = {}
      print(f"Top-1 Critical Attrs Comb: {criti_attr_dict}")
      
      # use influ_tuples_dict and criti_attr_dict to update the data (self.X_train, etc.)
      # at the same time, return an influence_vector for machine unlearning
      self.unlearning_vector = self.updateData(influ_tuples_dict, criti_attr_dict, iter_cnt)
      
      # save the results of this iter, used for multi-clean
      update_dataframe = pd.concat([self.X_train, self.y_train], axis=1)
      # update_dataframe.to_csv(os.path.join(self.dataset_rootdir, self.dataset_name, 'repaired', self.dataset_name + '_dirty_' + str(iter_cnt+1) + '.csv'), index = False)

      iter_cnt += 1

      
    # the model for proof
    if self.model_name != "FTTransformer":
      X_clean_embed, y_clean_embed = self.X_embed, self.y_embed
      self.train_clean_ds = getTrainDS(X_clean_embed, y_clean_embed, self.batch_size)
    else:
      self.train_clean_ds, _, _, _, _ = getTransformerDS(self.X_train, self.y_train, self.transformerpreprocessor, self.numerical_features, self.categorical_features, self.label_column, self.batch_size)
    
    # self.sub_model.fit(self.train_clean_ds, epochs=self.inc_epoch, verbose=0, shuffle = False)

    # the reported model
    if self.model_name == "FTTransformer":
      self.test_scores = self.model.predict(self.test_ds, verbose=0)
      eval_score = self.acc_dict['iter_' + str(iter_cnt-1)]
    else:
      self.test_scores = predictProb(self.X_test_embed, self.model)
      eval_score = evalScore(self.test_scores, self.y_test_embed)['F1']

    print("-" * 60)
    print("Rock4ML get the F1 score: " + str(eval_score))
    print("Dirty get the F1 score: " + str(self.dirty_score))
    print(self.cr_method + " get the F1 score: " + str(self.cr_score))
    
    return self.model
 
  def getPreprocessor(self, X_train, y_train):
    # get preprocessor through X and y train
    preprocessor = Preprocessor()
    preprocessor.fit(X_train, y_train)

    return preprocessor
  
  def transformData(self, preprocessor, X, y):
    # fit the data to get the embedding
    X_transform, y_transform = preprocessor.transform(X, y)

    X_transform = np.float64(X_transform)
    y_transform = np.float64(y_transform)

    return X_transform, y_transform

  def getInitMetric(self, pred, label):
    # get the evaluation metric, different metric use different calculation methods for init_metric
    # this init_metric is used for getCriticalAttr, and for validation set(coreset)

    if self.metric == 'binary_logloss':
      init_metric = log_loss(label, scipy.special.expit(pred), labels=[0, 1])
    elif self.metric == 'multi_logloss':
      init_metric = log_loss(label, scipy.special.softmax(pred, axis=1),
                              labels=list(range(pred.shape[1])))
    elif self.metric == 'rmse':
      init_metric = mean_squared_error(label, pred, squared=False)
    elif self.metric == 'auc':
      init_metric = roc_auc_score(label, scipy.special.expit(pred))
    else:
      raise NotImplementedError(f"Metric {self.metric} is not supported. "
                                f"Please select metric from ['binary_logloss', 'multi_logloss'"
                                f"'rmse', 'auc'].")
    return init_metric

  def cleanDataWithCR(self, iter_cnt, rock_port, multi_clean = False):
    ##### CRMethod Repair

    # if multi_clean is False, we will use one version of the repair data from start to end
    # if multi_clean is True, we repair the data each iter
    # for iter = iter_cnt, the repair data is stored in {dataset_rootdir}/{dataset_name}/rahabaran_tmp/{dataset_name}_{cr_method}_{iter_cnt}.csv
    # here if the repair file exists, we will read it, otherwise we will repair the data and store it

    repair_dataframe = None

    crresult_path = os.path.join(self.dataset_rootdir, self.dataset_name, 'repaired', self.dataset_name + '_repaired_' + self.cr_method + '.csv')
    
    if os.path.exists(crresult_path):
      repair_dataframe = pd.read_csv(crresult_path)
    else:
      cl = Cleaning(self.dataset_rootdir, self.dataset_name, self.label_column, verbose = True, labeling_budget = 20, multi_discovery=False, rock_port = rock_port)

      if self.cr_method == "rahabaran":
        repair_dataframe = cl.mixRahaBaran()
      elif self.cr_method == "rock":
        repair_dataframe = cl.rock()

      repair_dataframe.to_csv(crresult_path, index = False)
      repair_dataframe = pd.read_csv(crresult_path)

    return repair_dataframe

  def getCoreset(self, temperature, seed): 
    idxT, idxV = coreset_identification(self.X_train, self.acuml_losses, temperature, seed)
    return idxT, idxV

  def getMetric(self, y_train):
    if len(set(y_train.values.flatten())) > 2:
      self.metric = 'multi_logloss'
    elif len(set(y_train.values.flatten())) == 2:
      self.metric = 'binary_logloss'
    else:
      raise ValueError("Invalid metic.")
    
    return self.metric

  def getFeatures(self):
    dataframe = pd.concat([self.X_train, self.y_train], axis=1)
    cat_list = list(dataframe.select_dtypes(exclude=np.number))
    num_list = list(dataframe.select_dtypes(include=np.number))
    feat_list = dataframe.columns

    cat_list_no_label = [feature for feature in cat_list if feature != self.label_column]
    num_list_no_label = [feature for feature in num_list if feature != self.label_column]
    feat_list_no_label = [feature for feature in feat_list if feature != self.label_column]

    # get all features except the label

    return cat_list_no_label, num_list_no_label, feat_list_no_label

  def getInfluenceModel(self):
    if self.model_name == "FTTransformer":
      start_layer_name = "transformer-dense-last"
    else:
      start_layer_name = "last_layer"

    influence_model = InfluenceModel(
      self.model, 
      start_layer = start_layer_name,
      last_layer = None,
      loss_function=CategoricalCrossentropy(from_logits=False,reduction=Reduction.NONE) # always set the reduction to none when computing influences
    )
    
    ihvp_calculator = ExactIHVP(influence_model, self.train_ds.take(1))
    influence_calculator = FirstOrderInfluenceCalculator(influence_model, self.train_ds.take(1), ihvp_calculator, normalize=True)
    
    return influence_calculator

  def getInfluentialTuples(self, iter_cnt):
    if self.model_name != "FTTransformer":
      X_val_embed, y_val_embed = self.X_embed[self.idxV], self.y_embed[self.idxV]
    else:
      X_val_embed = y_val_embed = None
    
    tf.random.set_seed(self.random_state)
    if self.model_name == "FTTransformer":
      val_ds, _, _, _, y_val_embed = getTransformerDS(self.X_train.loc[self.idxV], self.y_train.loc[self.idxV], self.transformerpreprocessor, self.numerical_features, self.categorical_features, self.label_column, self.batch_size)
      val_losses = tf.keras.losses.categorical_crossentropy(y_val_embed, self.model(val_ds), from_logits=False).numpy().sum()    
    else:
      val_ds = None
      val_losses = tf.keras.losses.categorical_crossentropy(y_val_embed, self.model(X_val_embed, training=False).numpy(), from_logits=False).numpy().sum()
            
    self.influence_calculator = self.getInfluenceModel()

    influ_tuples_dict = {}

    if self.model_name != "FTTransformer":
      tuple_loader = tf.data.Dataset.from_tensor_slices((self.X_embed, self.y_embed))
      tuple_loader = tuple_loader.batch(self.X_embed.shape[0])  

      tuple_loader_repair = tf.data.Dataset.from_tensor_slices((self.X_repair_embed, self.y_repair_embed))
      tuple_loader_repair = tuple_loader_repair.batch(self.X_repair_embed.shape[0])  

    else:
      tuple_loader, _, _, _, _ = getTransformerDS(self.X_train, self.y_train, self.transformerpreprocessor, self.numerical_features, self.categorical_features, self.label_column, self.X_train.shape[0])
      tuple_loader_repair, _, _, _, _ = getTransformerDS(self.X_train_repair, self.y_train_repair, self.transformerpreprocessor, self.numerical_features, self.categorical_features, self.label_column, self.X_train_repair.shape[0])

    influe_vector_lists = influe_vector_cal(tuple_loader, self.influence_calculator)
    influe_vector_repair_lists = influe_vector_cal(tuple_loader_repair, self.influence_calculator)

    old_weights = self.model.get_weights()

    for idx in self.idxT:

      if self.X_train.loc[idx].equals(self.X_train_repair.loc[idx]):
        continue

      influe_vector = reshapeInfluevector(old_weights, influe_vector_lists[idx])
      influe_vector_repair = reshapeInfluevector(old_weights, influe_vector_repair_lists[idx]) 

      last_old_weights = old_weights[-2:]
      epsilon = 1 / len(self.X_train)
      
      last_new_weights = [last_old_weights[i] + epsilon * influe_vector[i] - epsilon * influe_vector_repair[i] for i in range(len(last_old_weights))]
      
      new_weights = copy.deepcopy(old_weights)
      new_weights[-2:] = last_new_weights

      self.model.set_weights(new_weights)

      if self.model_name != "FTTransformer":
        val_update_losses = tf.keras.losses.categorical_crossentropy(y_val_embed, self.model(X_val_embed, training=False).numpy(), from_logits=False).numpy().sum()
      else:
        val_update_losses = tf.keras.losses.categorical_crossentropy(y_val_embed, self.model(val_ds), from_logits=False).numpy().sum()

      if val_update_losses < val_losses:
        influ_tuples_dict[idx] = val_update_losses

    self.model.set_weights(old_weights)

    influ_tuples_dict = dict(sorted(influ_tuples_dict.items(), key=lambda item: item[1], reverse=False))
    
    return influ_tuples_dict
  
  def getCriticalAttrs(self):

    train_y = self.y_train_repair.loc[self.idxT]
    val_y = self.y_train.loc[self.idxV] # val data is coreset, we suppose it clean, so do not use y_train_repair

    train_init = self.init_scores.loc[self.idxT]
    val_init = self.init_scores.loc[self.idxV]
    init_metric = self.getInitMetric(val_init, val_y)

    avail_features = self.X_train_repair.columns.to_list()
    # initialize the candidate feature combinations
    candidate_feature_combs = [[feature] for feature in avail_features]
    # for candidate_feature, form a new data df_train_new = df_train_clean[candidate_feature] and set other column to none (make sure the shape if fixed from start to end)
    results = []
    combs_dict = {}

    iternum = 0
  
    train_x_raw = asType_copy(self.X_train, self.categorical_features).loc[self.idxT]
    val_x = asType_copy(self.X_train, self.categorical_features).loc[self.idxV] # coreset, suppose it clean, not use simul_X_train

    n_jobs = 1

    params = {"n_estimators": 100, "importance_type": "gain", "num_leaves": 16,
                        "random_state": self.random_state, "deterministic": True, "n_jobs": n_jobs, 'verbosity': -1}
    if self.metric is not None:
                  params.update({"metric": self.metric})
    
    gbm = lgb.LGBMClassifier(**params)
    
    gbm.fit(train_x_raw, train_y.values.ravel(), init_score=train_init,
                      eval_init_score=[val_init],
                      eval_set=[(val_x, val_y.values.ravel())],
                      callbacks=[lgb.early_stopping(3, verbose=False)])
    
    key_raw = list(gbm.best_score_['valid_0'].keys())[0]
    if self.metric in ['auc']:
      score_raw = gbm.best_score_['valid_0'][key_raw] - init_metric
    else:
      score_raw = init_metric - gbm.best_score_['valid_0'][key_raw]

    parallel_data = {'X_train': self.X_train, 'X_train_repair': self.X_train_repair, 
                      'clean_flag': self.clean_flag, 
                      'categorical_features': self.categorical_features, 
                      'idxT': self.idxT, 'idxV': self.idxV, 'metric': self.metric, 
                      'train_y': train_y, 'train_init': train_init, 'val_y': val_y, 'val_x': val_x,
                      'val_init': val_init, 'init_metric': init_metric, 'random_state': self.random_state}

    global_best_comb = ""
    global_best_score = float("-inf")

    while iternum < self.maxiter_ga:
      
      parallel_data['combs_dict'] = combs_dict

      # max_workers = min(96, self.maxoffspring_ga)
      max_workers = 1
      candidate_feature_combs_tuple = [tuple(element) for element in candidate_feature_combs]
      cal_features = list(set(candidate_feature_combs_tuple).difference(set(combs_dict.keys())))
      with ProcessPoolExecutor(max_workers = max_workers) as executor:
        futures = {executor.submit(process_candidate_attr, candidate_features, 
                                   parallel_data): candidate_features for candidate_features in cal_features}

      for future in as_completed(futures):
        candidate_features = futures[future]
        score = future.result()
        if score is not None:
          results.append([candidate_features, float("{:.8f}".format(score))])
          combs_dict[tuple(candidate_features)] = float("{:.8f}".format(score))

      candidate_features_scores = sorted(results, key=lambda x: (x[1], -len(x[0]), -len(x[0][0]), -ord(x[0][0][0])), reverse=True) 
      candidate_features_offsprings = updateCandidateFeatures(candidate_features_scores, self.topk_ga, 0.0, self.maxoffspring_ga, avail_features, self.random_state)
      candidate_feature_combs = candidate_feature_combs + candidate_features_offsprings
      
      if global_best_comb == candidate_features_scores[0][0] and iternum >= 5:
        print("GA early stop in iter", iternum+1)
        break

      if global_best_score < candidate_features_scores[0][1]:
        global_best_comb = candidate_features_scores[0][0]
        global_best_score = candidate_features_scores[0][1]

      
      iternum += 1

    sorted_combs_dict = dict(sorted(combs_dict.items(), key=lambda item: (item[1], -len(item[0]), -len(item[0][0]), -ord(item[0][0][0])), reverse=True))

    return sorted_combs_dict, score_raw
  
  def updateData(self, influ_tuples_dict, criti_attr_dict, iter_cnt):

    update_idx = set()

    old_tuples_X = []
    old_tuples_y = []

    X_train_embed_old = copy.deepcopy(self.X_embed)
    y_train_embed_old = copy.deepcopy(self.y_embed)

    # update influ_tuples
    indices_to_check = list(influ_tuples_dict.keys())
    rows_to_check = self.clean_flag.loc[indices_to_check]
    rows_to_update = self.X_train_repair.loc[indices_to_check]

    mask = rows_to_check == False

    self.X_train.loc[indices_to_check] = self.X_train.loc[indices_to_check].mask(mask, rows_to_update)
    self.clean_flag.loc[indices_to_check] = self.clean_flag.loc[indices_to_check].mask(mask, True)

    rows_updated = mask.any(axis=1)
    update_idx.update(set(rows_updated[rows_updated].index))
    update_influ_idx = copy.deepcopy(update_idx)

    # update critical attrs
    # only one attr_combs in criti_attr_dict.keys()

    for attr_combs in criti_attr_dict.keys():
      for attr in list(attr_combs):
        rows_to_update_criti = (self.clean_flag[attr] == False) & (self.clean_flag.index.isin(self.idxT) == True)
        self.X_train.loc[rows_to_update_criti, attr] = self.X_train_repair.loc[rows_to_update_criti, attr]
        self.clean_flag.loc[rows_to_update_criti, attr] = True

        update_idx.update(self.X_train[rows_to_update_criti].index.tolist())
    
    update_idx = list(update_idx)
    
    if self.model_name != "FTTransformer":
      # update X_embed, y_embed, train_ds
      self.X_embed, self.y_embed = self.transformData(self.preprocessor, self.X_train, self.y_train)
      self.train_ds = getTrainDS(self.X_embed, self.y_embed, self.batch_size)
    else:
      # update y_embed, train_ds
      self.train_ds, self.outputshape, self.categorical_column_dicts, _, self.y_embed = getTransformerDS(self.X_train, self.y_train, self.transformerpreprocessor, self.numerical_features, self.categorical_features, self.label_column, self.batch_size)
    
    if (len(update_influ_idx) == 0 and iter_cnt > 5) or len(update_idx) == 0 or (iter_cnt >= self.max_epoch_rock4ml):
      self.endflag = 1
      return None, None
    
    if self.model_name != "FTTransformer":
      for idx in update_idx:
        # these tuples have been updated
        old_tuples_X.append(X_train_embed_old[idx])
        old_tuples_y.append(y_train_embed_old[idx])

      old_tuples_X = np.array(old_tuples_X)
      old_tuples_y = np.array(old_tuples_y)

      tuple_loader = tf.data.Dataset.from_tensor_slices((old_tuples_X,old_tuples_y))
      tuple_loader = tuple_loader.batch(old_tuples_X.shape[0])

    else:
      tuple_loader, _, _, _, _ = getTransformerDS(self.X_train.loc[update_idx], self.y_train.loc[update_idx], self.transformerpreprocessor, self.numerical_features, self.categorical_features, self.label_column, batch_size=len(update_idx))

    influe_vector = influe_vector_cal(tuple_loader, self.influence_calculator)
    influe_vector = reshapeInfluevector(self.model_weights, influe_vector)

    del tuple_loader
    gc.collect()

    return influe_vector
  
  def identifyCleanData(self, pre_losses, iter_cnt):
    sub_losses = [pre_losses[i] - self.acuml_losses[i] for i in range(len(self.acuml_losses))]

    acuml_losses_without_clean = [loss for i, loss in enumerate(self.acuml_losses) if i not in self.idxClean]
    sub_losses_without_clean = [loss for i, loss in enumerate(sub_losses) if i not in self.idxClean]

    _, eta_1 = decide_eta_iqr(sub_losses_without_clean) # eta_1 is upper_bound
    eta_2, _ = decide_eta_iqr(acuml_losses_without_clean) # eta_2 is lower_bound

    cnt_a = cnt_b = 0

    for i in range(len(self.acuml_losses)):
      if i in self.idxClean: 
        continue

      if pre_losses[i] - self.acuml_losses[i] >= eta_1:
        self.idxClean = self.idxClean.union(pd.Index([i]))

        cnt_a += 1
      elif self.acuml_losses[i] <= eta_2 and iter_cnt > 1:
        self.idxClean = self.idxClean.union(pd.Index([i]))

        cnt_b += 1

    return self.idxClean
  
  def getTransformerPreprocessor(self, X_train, y_train):
    train_data = pd.concat([X_train, y_train], axis=1)
    # test_data = pd.concat([self.X_test, self.y_test], axis=1)

    NUMERIC_FEATURES = self.numerical_features
    CATEGORICAL_FEATURES = self.categorical_features
    
    FEATURES = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    LABEL = self.label_column

    label_enc = LabelEncoder()
    label_enc.fit(train_data[LABEL])
    train_data[LABEL] = label_enc.transform(train_data[LABEL])

    train_data[CATEGORICAL_FEATURES] = train_data[CATEGORICAL_FEATURES].astype(str)
    label_encoders = [NewLabelEncoder() for _ in range(len(CATEGORICAL_FEATURES))]
    train_data_categorical_features_np = train_data[CATEGORICAL_FEATURES].values
    for i in range(train_data_categorical_features_np.shape[1]):
      label_encoders[i].fit(train_data_categorical_features_np[:, i])

    train_data[NUMERIC_FEATURES] = train_data[NUMERIC_FEATURES].astype(float)
    sc = StandardScaler()
    if len(NUMERIC_FEATURES) != 0:
      sc.fit(train_data[NUMERIC_FEATURES])

    
    train_labels = train_data[[LABEL]].values
    
    encLabel = OneHotEncoder(sparse_output=False)
    encLabel.fit(train_labels)

    return {"label_enc": label_enc, "label_encoders": label_encoders, "sc": sc, "encLabel": encLabel}

  def getColumnType(self):
    columntype = {}
    for column in self.all_features:
      if column in self.numerical_features:
        columntype[column] = "float64"
      elif column in self.categorical_features:
        columntype[column] = "string"
      else:
        raise ValueError("Invalid column type.")
    return columntype

def getTransformerDS(X_data, y_data, transformer_preprocessor, numerical_features, categorical_features, label_column, batch_size, use_given = False, given_data = None):
  label_enc = transformer_preprocessor["label_enc"]
  label_encoders = transformer_preprocessor["label_encoders"]
  sc = transformer_preprocessor["sc"]
  encLabel = transformer_preprocessor["encLabel"]

  if use_given:
    train_data = pd.DataFrame(given_data).T
  else:
    train_data = pd.concat([X_data, y_data], axis=1)

  NUMERIC_FEATURES = numerical_features
  CATEGORICAL_FEATURES = categorical_features 
  FEATURES = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
  LABEL = label_column 

  train_data[LABEL] = label_enc.transform(train_data[LABEL])
  
  train_data[CATEGORICAL_FEATURES] = train_data[CATEGORICAL_FEATURES].astype(str)
  train_data_categorical_features_np = train_data[CATEGORICAL_FEATURES].values
  train_transformed_categorical_features = [label_encoders[i].transform(train_data_categorical_features_np[:, i]) for
                                      i in range(train_data_categorical_features_np.shape[1]) ]

  categorical_column_dicts = []
  for i in range(train_data_categorical_features_np.shape[1]):
    cdict = {}
    for x, y in zip(train_transformed_categorical_features[i], train_data_categorical_features_np[:, i]):
      cdict[x] = y
    categorical_column_dicts.append(cdict)

  for i, cname in enumerate(CATEGORICAL_FEATURES):
    train_data[cname] = train_transformed_categorical_features[i]

  train_data[CATEGORICAL_FEATURES] = train_data[CATEGORICAL_FEATURES].astype(float)
  train_data[NUMERIC_FEATURES] = train_data[NUMERIC_FEATURES].astype(float)
  if len(NUMERIC_FEATURES) != 0:
    train_data.loc[:, NUMERIC_FEATURES] = sc.transform(train_data[NUMERIC_FEATURES])

  train_labels = train_data[[LABEL]].values
  
  train_labels_onehot = encLabel.transform(train_labels)
  train_dataset = df_to_dataset_np_cat_label(train_data[FEATURES + [LABEL]], train_labels_onehot, LABEL, batch_size=batch_size)
  label_card = np.unique(train_labels).shape[0]

  return train_dataset, label_card, categorical_column_dicts, train_data[FEATURES], train_labels_onehot
      
def process_candidate_attr(candidate_features, data):
  
  X_train = data['X_train']
  X_train_repair = data['X_train_repair']
  categorical_features = data['categorical_features']
  idxT = data['idxT']
  idxV = data['idxV']
  metric = data['metric']
  train_y = data['train_y']
  train_init = data['train_init']
  val_y = data['val_y']
  val_x = data['val_x']
  val_init = data['val_init']
  init_metric = data['init_metric']
  random_state = data['random_state']
  
  simul_X_train = X_train_repair.copy()
  
  for column in simul_X_train.columns:
    # for not candidate, change back to dirty
    if column not in candidate_features:
      simul_X_train[column] = X_train[column]

  simul_X_train = asType(simul_X_train, categorical_features)
  train_x = simul_X_train.loc[idxT]

  #### X_train or simul_X_train ?
  # val_x = asType(X_train, categorical_features).loc[idxV] # coreset, suppose it clean, not use simul_X_train

  n_jobs = 4

  params = {"n_estimators": 100, "importance_type": "gain", "num_leaves": 16,
                      "random_state": random_state, "deterministic": True, "n_jobs": n_jobs, 'verbosity': -1}
  if metric is not None:
                params.update({"metric": metric})
  
  gbm = lgb.LGBMClassifier(**params)
  
  gbm.fit(train_x, train_y.values.ravel(), init_score=train_init,
                    eval_init_score=[val_init],
                    eval_set=[(val_x, val_y.values.ravel())],
                    callbacks=[lgb.early_stopping(3, verbose=False)])
  
  key = list(gbm.best_score_['valid_0'].keys())[0]
  if metric in ['auc']:
    score = gbm.best_score_['valid_0'][key] - init_metric
  else:
    score = init_metric - gbm.best_score_['valid_0'][key]

  return score

def load_model(model_name, X_embed, y_embed, optimizer, 
               random_state, numerical_features = None, 
               categorical_features = None,
               categorical_column_dicts = None, X_train = None,
               label_card = None):
  
  if model_name != "FTTransformer":
    if y_embed.shape[1] > 2:
      last_activation = 'softmax'
    else:
      last_activation = 'sigmoid'

  if model_name == "LogisticRegression":
    tf.random.set_seed(random_state)
    model = Sequential()
    model.add(Dense(y_embed.shape[1], input_shape=(X_embed.shape[1],), activation=last_activation, name='last_layer'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
  
  elif model_name == "MLPClassifier":
    tf.random.set_seed(random_state)
    model = Sequential()
    model.add(Dense(32, input_shape=(X_embed.shape[1],), activation='tanh'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(y_embed.shape[1], activation=last_activation, name='last_layer'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

  elif model_name == "L2SVM":
    tf.random.set_seed(random_state)
    model = Sequential()
    model.add(Dense(y_embed.shape[1], input_shape=(X_embed.shape[1],),
                    activation=None, kernel_regularizer=tf.keras.regularizers.l2(0.1), 
                    name = "last_layer"))
    
    model.compile(optimizer=optimizer, loss=l2_svm_loss, metrics=['accuracy'])
    
  elif model_name == "FTTransformer":
    tf.random.set_seed(random_state)

    ft_linear_encoder = FTTransformerEncoder(
      numerical_features = numerical_features,
      categorical_features = categorical_features,
      numerical_data = X_train[numerical_features].values,
      categorical_data = np.array(X_train[categorical_features].values, 'str'),
      y = None,
      numerical_embedding_type='linear',
      embedding_dim=8,
      depth=1,
      heads=2,
      attn_dropout=0.2,
      ff_dropout=0.2,
      categorical_column_dicts=categorical_column_dicts,
      explainable=False
    )

    model = FTTransformer(
      encoder=ft_linear_encoder,
      out_dim=label_card,
      out_activation='softmax',
    )

    model.compile(
      optimizer = optimizer,
      loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE),
      metrics= tf.keras.metrics.AUC(name="PR AUC", curve='PR')
    )

  return model

def trainModel(train_ds, X_embed, y_embed, model_name, optimizer, no_epoch, 
               random_state, numerical_features = None, 
               categorical_features = None,
               categorical_column_dicts = None, X_train = None,
               label_card = None):
  
  model = load_model(model_name, X_embed, y_embed, optimizer, 
                     random_state, numerical_features, categorical_features,
                     categorical_column_dicts, X_train, label_card)
  
  sub_model = load_model(model_name, X_embed, y_embed, optimizer, 
                     random_state, numerical_features, categorical_features,
                     categorical_column_dicts, X_train, label_card)
  
  model.fit(train_ds, epochs=no_epoch, verbose=0, shuffle = False)

  C = model
  
  # return the losses of the last epoch when first training
  return C, sub_model, optimizer

def predictProb(X_embed, model):
  return model.predict(X_embed, verbose=0)
  
def influe_vector_cal(tuple_in_loader, influe_calculator):
    
  inf_vector = influe_calculator.compute_influence_vector(tuple_in_loader)

  for sample_id, ((sample, label), inf_val_new) in inf_vector.enumerate():
    I_up_new_vector = inf_val_new.numpy()

  return I_up_new_vector

def getTrainDS(X, y, batch_size):
  tensordata = tf.data.Dataset.from_tensor_slices((X, y))
  train_ds = tensordata.batch(batch_size)
  return train_ds

def evalScore(pred_scores, y_embed):
  y_pred = np.argmax(pred_scores, axis=1)
  y_test = np.argmax(y_embed, axis=1)

  try:
    score = {'Precision': precision_score(y_true = y_test, y_pred = y_pred, average ='macro'),
             'Recall': recall_score(y_true = y_test, y_pred = y_pred, average = 'macro'),
             'F1': f1_score(y_true = y_test, y_pred = y_pred, average = 'macro')}
  except:
    score = {'Precision': -1, 'Recall': -1, 'F1': -1}

  return score

def reshapeInfluevector(model_weights, influe_vector):
  influe_vector_flatten = influe_vector.flatten()
  x = 0
  inf_weight_reshape = []
  for i in model_weights[-2:]:
    i_copy = i.flatten()
    y = x + i_copy.shape[0]
    vec_slice = influe_vector_flatten[x:y]
    vec_slice = np.reshape(vec_slice, i.shape)
    # print(vec_slice.shape)
    x = y
    
    inf_weight_reshape.append(vec_slice)
  
  return inf_weight_reshape

def asType(dataframe, categorical_features):
  for feature in categorical_features:
    dataframe[feature] = dataframe[feature].astype('category')
    dataframe[feature] = dataframe[feature].cat.codes
    dataframe[feature] = dataframe[feature].astype('category')
  return dataframe

def asType_copy(dataframe, categorical_features):
  dataframe_new = copy.deepcopy(dataframe)
  for feature in categorical_features:
    dataframe_new[feature] = dataframe_new[feature].astype('category')
    dataframe_new[feature] = dataframe_new[feature].cat.codes
    dataframe_new[feature] = dataframe_new[feature].astype('category')
  return dataframe_new

def decide_eta_iqr(data):
  quartile_1, quartile_3 = np.percentile(data, [25, 75])
  iqr = quartile_3 - quartile_1
  
  lower_bound = quartile_1 - (iqr * 1.5)
  upper_bound = quartile_3 + (iqr * 1.5)
  
  return lower_bound, upper_bound

class NewLabelEncoder(LabelEncoder):
  def transform(self, y):
    unseen_labels = set(y) - set(self.classes_)
    if unseen_labels:
      self.classes_ = np.concatenate((self.classes_, list(unseen_labels)))
      self.n_classes_ = len(self.classes_)
  
    return super().transform(y)