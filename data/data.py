# Helpful links:
# https://stackoverflow.com/questions/46401209/how-to-convert-the-arff-object-loaded-from-a-arff-file-into-a-dataframe-format
#https://stackoverflow.com/questions/40389764/how-to-translate-bytes-objects-into-literal-strings-in-pandas-dataframe-pytho
# https://stackoverflow.com/questions/4056768/how-to-declare-array-of-zeros-in-python-or-an-array-of-a-certain-size

import numpy as np
import pandas as pd
from scipy.io.arff import loadarff 
import random as rand
import os
from io import StringIO


activity = 'A'  # Code for walking activity
number_subjects = 50  # Number of ARFF files per sensor (1 per subject)
all_data = pd.DataFrame()  # Pandas dataframe to hold all subject data for the 1 activity
stats = [[0] for row in range(number_subjects)]  # Holds stats from model testing
# print(len(stats))

# Open folder of data for phone accel sensor and process files one-by-one
for dirpath, dirnames, files in os.walk('arff/phone/accel'):
  temp_data = pd.DataFrame()
  files.sort()
  for file_name in files:
    print(file_name)
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

    temp_data = pd.concat([temp_data, df_data], ignore_index=True)  # Accumulate accel data
  
  # print(temp_data)
  all_data = pd.concat([all_data, temp_data])  # Add all accel data to main dataframe

print(all_data)

# Open folder of data for phone gyro sensor and process files one-by-one
for dirpath, dirnames, files in os.walk('arff/phone/gyro'):
  temp_data = pd.DataFrame()
  files.sort()
  for file_name in files:
    print(file_name)
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

    temp_data = pd.concat([temp_data, df_data], ignore_index=True)  # Accumulate accel data
  
  # print(temp_data)
  all_data = pd.concat([all_data, temp_data], axis=1)  # Add all accel data to main dataframe

print(all_data)

# Looks right, but add code to verify a_class = g_class for all rows
# WIP...