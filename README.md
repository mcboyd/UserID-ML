# UserID-ML
User identification through activity classification from smartphone and smart-wearable sensor data using machine learning. Utilizing the "[WISDM Smartphone and Smartwatch Activity and Biometrics Dataset](https://archive.ics.uci.edu/ml/datasets/WISDM+Smartphone+and+Smartwatch+Activity+and+Biometrics+Dataset+)".

### Data Extraction
Code, pseudocode, and concepts documented in [data folder](data/README.md).  
- Authorized Person: 90 seconds of data for each subject/activity is allocated to each of the testing and training data sets
- Imposter: 18 other subjects are randomly selected and 30 seconds of data for the same activity are randomly chosen for each subject
  - Data from 9 subjects is placed in the testing data set
  - While data from the other 9 is placed in the training data set
- Training class ratio is 90:270 (subject:imposter) by design

### Random Forest Algorithm
Will use the TensorFlow class BoostedTreesClassifier as the model. Requirements:
- Separate features and labels with Dataframe.pop()
- Convert dataframe into Tensor with tf.data.Dataset.from_tensor_slices()
- Define input function to break data into batches
- Define evaluate function to run on test data
- Create model, train, and evaluate
