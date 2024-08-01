import keras
from keras import layers
# from keras import ops
# from keras.layers.core import ops
import tensorflow as tf
import math
import numpy as np
import pandas as pd
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
from functools import partial
from tensorflow import data as tf_data

def transform_element(elem, transform_list):
  for i in range(len(transform_list)):
    if elem == transform_list[i]:
      return i
  
  raise ValueError("Not in Dict")

def getTabTransInput(X, cat_dict):
  X_copy = X.copy()
  for col in X.columns:
    if col in cat_dict.keys():  # 只处理在cat_lists字典中有对应映射的列
      # 将每个元素替换成cat_lists字典中的映射
      X_copy[col] = X_copy[col].apply(lambda x: transform_element(x, cat_dict[col]))
  return X_copy

def create_model_inputs(feature_names, numeric_feature_names):
  inputs = {}
  for feature_name in feature_names:
    if feature_name in numeric_feature_names:
      inputs[feature_name] = layers.Input(
        name=feature_name, shape=(), dtype="float32"
      )
    else:
      inputs[feature_name] = layers.Input(
        name=feature_name, shape=(), dtype="int32"
      )
  return inputs

def encode_inputs(inputs, embedding_dims, categorical_feature_names, categorical_features_with_vocabulary):
  encoded_categorical_feature_list = []
  numerical_feature_list = []

  for feature_name in inputs:
    if feature_name in categorical_feature_names:
      vocabulary = categorical_features_with_vocabulary[feature_name]
      # Create a lookup to convert a string values to an integer indices.
      # Since we are not using a mask token, nor expecting any out of vocabulary
      # (oov) token, we set mask_token to None and num_oov_indices to 0.

      # Convert the string input values into integer indices.

      # Create an embedding layer with the specified dimensions.
      embedding = layers.Embedding(
          input_dim=len(vocabulary), output_dim=embedding_dims
      )

      # Convert the index values to embedding representations.
      encoded_categorical_feature = embedding(inputs[feature_name])
      encoded_categorical_feature_list.append(encoded_categorical_feature)

    else:
      # Use the numerical features as-is.
      numerical_feature = tf.expand_dims(inputs[feature_name], -1)
      numerical_feature_list.append(numerical_feature)

  return encoded_categorical_feature_list, numerical_feature_list

def create_mlp(hidden_units, dropout_rate, activation, normalization_layer, name=None):
  mlp_layers = []
  for units in hidden_units:
    mlp_layers.append(normalization_layer())
    mlp_layers.append(layers.Dense(units, activation=activation))
    mlp_layers.append(layers.Dropout(dropout_rate))

  return keras.Sequential(mlp_layers, name=name)

def create_tabtransformer_classifier(
  num_transformer_blocks,
  num_heads,
  embedding_dims,
  mlp_hidden_units_factors,
  dropout_rate,
  feature_names,
  numeric_feature_names,
  categorical_feature_names,
  categorical_features_with_vocabulary,
  output_shape,
  use_column_embedding=False,
):
  # Create model inputs.
  inputs = create_model_inputs(feature_names, numeric_feature_names)
  # encode features.
  encoded_categorical_feature_list, numerical_feature_list = encode_inputs(
    inputs, embedding_dims, categorical_feature_names, categorical_features_with_vocabulary
  )
  # Stack categorical feature embeddings for the Transformer.
  encoded_categorical_features = tf.stack(encoded_categorical_feature_list, axis=1)
  # Concatenate numerical features.
  numerical_features = layers.concatenate(numerical_feature_list)

  # Add column embedding to categorical feature embeddings.
  if use_column_embedding:
    num_columns = encoded_categorical_features.shape[1]
    column_embedding = layers.Embedding(
      input_dim=num_columns, output_dim=embedding_dims
    )
    column_indices = tf.arange(start=0, stop=num_columns, step=1)
    encoded_categorical_features = encoded_categorical_features + column_embedding(
      column_indices
    )

  # Create multiple layers of the Transformer block.
  for block_idx in range(num_transformer_blocks):
    # Create a multi-head attention layer.
    attention_output = layers.MultiHeadAttention(
      num_heads=num_heads,
      key_dim=embedding_dims,
      dropout=dropout_rate,
      name=f"multihead_attention_{block_idx}",
    )(encoded_categorical_features, encoded_categorical_features)
    # Skip connection 1.
    x = layers.Add(name=f"skip_connection1_{block_idx}")(
      [attention_output, encoded_categorical_features]
    )
    # Layer normalization 1.
    x = layers.LayerNormalization(name=f"layer_norm1_{block_idx}", epsilon=1e-6)(x)
    # Feedforward.
    feedforward_output = create_mlp(
      hidden_units=[embedding_dims],
      dropout_rate=dropout_rate,
      activation=keras.activations.gelu,
      normalization_layer=partial(
        layers.LayerNormalization, epsilon=1e-6
      ),  # using partial to provide keyword arguments before initialization
      name=f"feedforward_{block_idx}",
    )(x)
    # Skip connection 2.
    x = layers.Add(name=f"skip_connection2_{block_idx}")([feedforward_output, x])
    # Layer normalization 2.
    encoded_categorical_features = layers.LayerNormalization(
      name=f"layer_norm2_{block_idx}", epsilon=1e-6
    )(x)

  # Flatten the "contextualized" embeddings of the categorical features.
  categorical_features = layers.Flatten()(encoded_categorical_features)
  # Apply layer normalization to the numerical features.
  numerical_features = layers.LayerNormalization(epsilon=1e-6)(numerical_features)
  # Prepare the input for the final MLP block.
  features = layers.concatenate([categorical_features, numerical_features])

  # Compute MLP hidden_units.
  mlp_hidden_units = [
      factor * features.shape[-1] for factor in mlp_hidden_units_factors
  ]
  # Create final MLP.
  features = create_mlp(
      hidden_units=mlp_hidden_units,
      dropout_rate=dropout_rate,
      activation=keras.activations.selu,
      normalization_layer=layers.BatchNormalization,
      name="MLP",
  )(features)

  # Add a sigmoid as a binary classifer.
  outputs = layers.Dense(units=output_shape, activation="sigmoid", name='last_layer')(features)
  model = keras.Model(inputs=inputs, outputs=outputs)
  return model

def create_tabtransformer_optimizer(learning_rate=0.001, weight_decay=0.0001):
  optimizer = keras.optimizers.AdamW(
    learning_rate=learning_rate, weight_decay=weight_decay
  )
  return optimizer

# tabtransformer_model = create_tabtransformer_classifier(
#   num_transformer_blocks=3,
#   num_heads=4,
#   embedding_dims=16,
#   mlp_hidden_units_factors=[2,1,],
#   dropout_rate=0.2,
# )

# print("Total model weights:", tabtransformer_model.count_params())
# keras.utils.plot_model(tabtransformer_model, show_shapes=True, rankdir="LR")