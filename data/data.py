# Helpful links:
# https://stackoverflow.com/questions/46401209/how-to-convert-the-arff-object-loaded-from-a-arff-file-into-a-dataframe-format
#https://stackoverflow.com/questions/40389764/how-to-translate-bytes-objects-into-literal-strings-in-pandas-dataframe-pytho
# https://stackoverflow.com/questions/4056768/how-to-declare-array-of-zeros-in-python-or-an-array-of-a-certain-size
# https://stackoverflow.com/questions/33907776/how-to-create-an-array-of-dataframes-in-python
# https://blog.tensorflow.org/2019/03/how-to-train-boosted-trees-models-in-tensorflow.html
# https://ai.stackexchange.com/questions/6383/what-do-prediction-mean-and-label-mean-represent-in-this-tensorflow-code?rq=1

import math
import pickle
import numpy as np
import pandas as pd
from scipy.io.arff import loadarff 
import random as rand
import os
import sys
from io import StringIO
import tensorflow as tf
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# Use while debugging - gave me an error...
# tf.enable_eager_execution()


activity = 'A'  # Code for walking activity
number_subjects = 44  # Number of ARFF files per sensor (1 per subject)
subjects = []  # Array to hold list of subject/class ids
all_data = {}  # Collection to hold dataframes of all subjects' data for the 1 activity
stats = [[0] for row in range(number_subjects)]  # Holds stats from model testing

def process_data_files():
  dataframe_collection = {}
  # Open folder of data for phone accel sensor and process files one-by-one
  for dirpath, dirnames, files in os.walk('arff/phone/accel'):
    files.sort()  # Sort to force them to process in alphabetical order
    for file_name in files:
      raw_data, meta = loadarff(os.path.join(dirpath, file_name))
      df_data = pd.DataFrame(raw_data)  # Import arff data into dataframe

      # 2 of the dataframe columns import as "bytes"; below 4 lines convert them to strings
      str_df = df_data.select_dtypes([np.object]) 
      str_df = str_df.stack().str.decode('utf-8').unstack()
      for col in str_df:
          df_data[col] = str_df[col]
      
      df_data = df_data[df_data.ACTIVITY == activity]  # Remove rows for all other activities
      df_data = df_data.add_prefix('a_')  # Prefix column labels to identify as accel data 

      for col in df_data.iloc[:,43:91]:  # Delete columns of unused data
        df_data = df_data.drop(columns=[col])
      df_data = df_data.drop(columns=['a_ACTIVITY'])  # Delete ACTIVITY column (unused)
      subj_id = df_data.a_class[0]

      # Add cleaned-up accel data to dataframe collection
      dataframe_collection[subj_id] = df_data.copy()

  # Open folder of data for phone gyro sensor and process files one-by-one
  for dirpath, dirnames, files in os.walk('arff/phone/gyro'):
    files.sort()  # Sort to force them to process in alphabetical order
    for file_name in files:
      raw_data, meta = loadarff(os.path.join(dirpath, file_name))
      df_data = pd.DataFrame(raw_data)  # Import arff data into dataframe

      # 2 of the dataframe columns import as "bytes"; below 4 lines convert them to strings
      str_df = df_data.select_dtypes([np.object]) 
      str_df = str_df.stack().str.decode('utf-8').unstack()
      for col in str_df:
          df_data[col] = str_df[col]
      
      df_data = df_data[df_data.ACTIVITY == activity]  # Remove rows for all other activities
      df_data = df_data.add_prefix('g_')  # Prefix column labels to identify as gyro data

      for col in df_data.iloc[:,43:91]:  # Delete columns of unused data
        df_data = df_data.drop(columns=[col])
      df_data = df_data.drop(columns=['g_ACTIVITY'])  # Delete ACTIVITY column (unused)
      subj_id = df_data.g_class[0]
      subjects.append(subj_id)  # Add subject/class ids to subjects array

      # Horizontally append cleaned-up gyro data to dataframe collection items
      dataframe_collection[subj_id] = pd.concat([dataframe_collection[subj_id], df_data.copy()], axis=1)

  # Finalize data (should be 87 columns wide w/ g_class, the class/category, as last column): 
  # 1. Remove rows with empty data (where accel and gyro data had different number of rows)
  # 2. Remove duplicate class from middle column
  for key in dataframe_collection.keys():
    dataframe_collection[key].dropna(inplace=True)
    dataframe_collection[key].drop(columns=['a_class'], inplace=True)
    all_data[key] = dataframe_collection[key].copy()
      
# Generate imposter data for training: 
# 18 other subjects are randomly selected (not including passed-in "subject_id")
# 30 seconds of data for the same activity are randomly chosen for each "other" subject
def gen_imposter(subject_id):
  imp_list = []
  temp_list = subjects.copy()
  temp_list.remove(subject_id)
  imp_list = rand.sample(temp_list, 18)
  imp_data = pd.DataFrame()  # 18 subjects, 30 secs data/subject; each row=10 secs => 3 rows/subject
  for imposter in imp_list:
    df1 = all_data[imposter].sample(3)
    imp_data = pd.concat([imp_data, df1.copy()], ignore_index=True)
  imp_data.g_class = '0'  # Set the subject id column for all of the imposter data to be 0; this is their "class"
  return imp_data

# Defines a function that returns (features, labels) as required for TF train() and evaluate()
def input_fn(subject, imposter, batch_size, shuffle=True, n_epochs=None):
  data = pd.concat([subject, imposter]) # Combine subject and impostor data
  labels = np.sign(data.pop('g_class').astype('float32')) # Use binary classifiers, 0 for imposter 1 for subject
  dataset = tf.data.Dataset.from_tensor_slices((dict(data), labels)) # Convert to Tensor
  if shuffle:
    dataset = dataset.shuffle(len(data.index))
  dataset = dataset.repeat(n_epochs) # n_epochs=None will repeat until training is done
  return dataset.batch(batch_size)

def train(subject, imposter):
  feature_columns = []
  for feature_name in subject.columns:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
  feature_columns.pop() # Class should not be part of the feature columns
  batch = math.floor(math.sqrt(len(subject.index)+len(imposter.index))) # Use batch size sqrt(N)
  model = tf.estimator.BoostedTreesClassifier(feature_columns, n_batches_per_layer=batch)
  model.train(input_fn=lambda: input_fn(subject, imposter, batch), max_steps=100)
  return model


print("Processing data files...")
process_data_files()
pickle.dump(all_data, open("checkpoints/all_data.p", 'wb'))
pickle.dump(subjects, open("checkpoints/subjects.p", 'wb'))

#print("Loading data and subjects from pickle files")
#all_data = pickle.load(open("pickle/all_data.p", "rb"))
#subjects = pickle.load(open("pickle/subjects.p", "rb"))

# Now do the rest of the work (train, test, stats)
idx = 0
for subject in subjects:
  df1 = all_data[subject].copy()  # Grab subject data and store temporarily
  subject_train = df1.sample(9)  # Grab the training data
  df1 = df1.drop(subject_train.index)  # Remove the training data from temp storage
  subject_test = df1.copy()  # Grab the testing data (remainder)
  
  # Get imposter data that excludes the current subject
  imposter = gen_imposter(subject)
  imp_train = imposter.sample(27)  # Take 27 random rows of imposter data for training
  imposter = imposter.drop(imp_train.index)  # Remove the training data
  imp_test = imposter.copy()  # And the rest of the imposter data for testing

  # Train model
  print("\nTraining model for subject ", subject, "...")
  model = train(subject_train, imp_train)

  # Test model
  print("\nTesting model for subject ", subject, "...")
  n_samples = len(subject_test.index) + len(imp_test.index)
  stats[idx] = model.evaluate(input_fn=lambda: input_fn(subject_test, imp_test, n_samples, False, 1))
  idx += 1

y_true = [0] * 44
y_true[0] = 1
y_score = []
for i in range(number_subjects):
  print("\nStats for subject ", subjects[i], ":")
  print(pd.Series(stats[i]))
  print(stats[i])
  #y_true.append(stats[i].get('label/mean'))
  y_score.append(stats[i].get('prediction/mean'))

# Calculate the EER
fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
#fnr = 1 - tpr
#eer = fpr(np.nanargmin(np.absolute((fnr - fpr))))
print("\nEER: ", eer)

pickle.dump(stats, open("checkpoints/stats.p", 'wb'))

print("\nDone.")
