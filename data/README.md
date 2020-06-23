## UserID-ML: Data
Code, pseudocode, and concepts for extracting and transforming data for UserID-ML project.


### Concepts
See [data_pseudocode.py](data_pseudocode.py).
- Authorized Person: 90 seconds of data for each subject/activity is allocated to each of the testing and training data sets
- Imposter: 18 other subjects are randomly selected and 30 seconds of data for the same activity are randomly chosen for each subject
  - Data from 9 subjects is placed in the testing data set
  - While data from the other 9 is placed in the training data set
- Training class ratio is 90:270 (subject:imposter) by design  

***NOTE:** Pseudocode file has not been updated since actual coding started. It was just a starting point.*


### "Raw" data - ARFF files
The [ARFF files](arff) contain the aggregated and binned data as supplied by the paper authors.   
Unfortunately the data in the files cannot be imported to a Pandas dataframe as-is. There are some minor formatting tweaks required:
- Remove double-quotes around each column name *(these prevent the column names from importing into the dataframe)*
- Remove extra spaces in column definitions where column data is non-numeric *(these prevent the associated column data from importing)*  

As I am on a Windows PC, I wrote a small [Powershell script](arff_fixes.ps1) to batch-update the files for me.  
**NOTE:** There are 51 subjects listed in the paper, but no ARFF files for subjects #11-17, so we're only looking at 44 subjects now.


### Import Data
Ongoing work in file [data.py](data.py).
There is a **SciPy** function to import ARFF files. The files are first imported into an array using this function, then a Pandas dataframe is created from the array.  
Columns containing string data are imported as bytes, though, and their data must be recast as strings. This is the next step (there are 2 columns like this), and then the data is ready for manipulation.  

The number of rows of data for each subject can differ between the sensors (e.g., 17 rows of data for the accel, but 22 rows of data for the gyro).  To get around this limitation we:
1. Combine the accel and gyro data for each subject
   - Combination is veritcal so 44 columns of data per sensor become 88 columns of combined data
2. Remove all rows where there are missing values 

The combined and trimmed data for each subject is stored in a Pandas dataframe. Once all of the dataframes are created, we iterate through them one-by one and split them into test and train data.  
The goal, as stated in the paper, was to put 90 seconds of data per subject into each of the test and train data sets. Unfortunately not all subjects have 180 seconds of data. We chose to put 90 seconds into training and the remainder, whatever it may be, into testing.   


### Manipulate Data
Final dataframe will have 87 columns = (43 data x 2) + subject_id (the "class", as last column)
