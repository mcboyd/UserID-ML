# UserID-ML
User identification through activity classification from smartphone and smart-wearable sensor data using machine learning. Utilizing the "[WISDM Smartphone and Smartwatch Activity and Biometrics Dataset](https://archive.ics.uci.edu/ml/datasets/WISDM+Smartphone+and+Smartwatch+Activity+and+Biometrics+Dataset+)".

### Data Extraction
Code, pseudocode, and concepts documented in [data folder](data).  
- Authorized Person: 90 seconds of data for each subject/activity is allocated to each of the testing and training data sets
- Imposter: 18 other subjects are randomly selected and 30 seconds of data for the same activity are randomly chosen for each subject
  - Data from 9 subjects is placed in the testing data set
  - While data from the other 9 is placed in the training data set
- Training class ratio is 90:270 (subject:imposter) by design

<<<<<<< HEAD
### Equal Error Rate (EER)
Pseudocode in [equal_error_rate.py](equal_error_rate.py).
- Technical documentation and formulas: https://www.researchgate.net/post/Any_advice_on_computing_Equal_Error_Rate
- We can use ROC from the sklearn library
  - https://stackoverflow.com/questions/28339746/equal-error-rate-in-python
  - https://yangcha.github.io/EER-ROC/
- ROC (Receiver Operating Characteristic)
  - https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5
  - https://www.statisticshowto.com/receiver-operating-characteristic-roc-curve/
- Plot the true positive rate against the false positive rate to get the ROC, then use the sklearn library to calculate the EER

=======
### Random Forest Algorithm
Will use the TensorFlow class BoostedTreesClassifier as the model. Requirements:
- Separate features and labels with Dataframe.pop()
- Convert dataframe into Tensor with tf.data.Dataset.from_tensor_slices()
- Define input function to break data into batches
- Define evaluate function to run on test data
- Create model, train, and evaluate
>>>>>>> 42aa87382693ce50e8b2d279a07312d89170ef71
