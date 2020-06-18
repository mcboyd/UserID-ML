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
  # print(f'Found directory: {dirpath}')
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
    df_data = df_data.add_prefix('a_')
    # print(df_data)
    # print(df_data.X3)

    for col in df_data.iloc[:,1:43]:  # Extract first 42 columns of data
      all_data[col] = df_data[col].copy()

    # Exatrct 43rd column of data
    all_data = pd.concat([all_data, df_data.a_RESULTANT], axis=1).copy()
    
    print(all_data)

# Open folder of data for phone gyro sensor and process files one-by-one
for dirpath, dirnames, files in os.walk('arff/phone/gyro'):
  # print(f'Found directory: {dirpath}')
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
    df_data = df_data.add_prefix('g_')
    # print(df_data)
    # print(df_data.X3)

    for col in df_data.iloc[:,1:43]:  # Extract first 42 columns of data
      all_data[col] = df_data[col].copy()

    # Exatrct 43rd column of data
    all_data = pd.concat([all_data, df_data.g_RESULTANT], axis=1).copy()
    
    print(all_data)


# WIP...

