# UserID-ML
User identification through activity classification from smartphone and smart-wearable sensor data using machine learning. Utilizing the "[WISDM Smartphone and Smartwatch Activity and Biometrics Dataset](https://archive.ics.uci.edu/ml/datasets/WISDM+Smartphone+and+Smartwatch+Activity+and+Biometrics+Dataset+)".

### Data Extraction
Pseudocode and concepts documented in [data_pseudocode.py](data_pseudocode.py).  
- Authorized Person: 90 seconds of data for each subject/activity is allocated to each of the testing and training data sets
- Imposter: 18 other subjects are randomly selected and 30 seconds of data for the same activity are randomly chosen for each subject
  - Data from 9 subjects is placed in the testing data set
  - While data from the other 9 is placed in the training data set
- Training class ratio is 90:270 (subject:imposter) by design
