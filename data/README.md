## UserID-ML: Data
Code, pseudocode, and concepts for extracting and transforming data for UserID-ML project.


### Concepts
See [data_pseudocode.py](data_pseudocode.py).
- Authorized Person: 90 seconds of data for each subject/activity is allocated to each of the testing and training data sets
- Imposter: 18 other subjects are randomly selected and 30 seconds of data for the same activity are randomly chosen for each subject
  - Data from 9 subjects is placed in the testing data set
  - While data from the other 9 is placed in the training data set
- Training class ratio is 90:270 (subject:imposter) by design


### "Raw" data - ARFF files
The [ARFF files](arff) contain the aggregated and binned data as supplied by the paper authors.   
Unfortunately the data in the files cannot be imported to a Pandas dataframe as-is. There are some minor formatting tweaks required:
- Remove double-quotes around each column name *(these prevent the column names from importing into the dataframe)*
- Remove extra spaces in column definitions where column data is non-numeric *(these prevent the associated column data from importing)*  
As I am on a Windows PC, I wrote a small [Powershell script](arff_fixes.ps1) to batch-update the files for me.  
**NOTE:** There are 51 subjects listed in the paper, but no ARFF files for subject #14, so we're only looking at 50 subjects for now.


### Import Data
Ongoing work in file [data.py](data.py).
There is a **SciPy** function to import ARFF files. The files are first imported into an array using this function, then a Pandas dataframe is created from the array.  
Columns containing string data are imported as bytes, though, and their data must be recast as strings. This is the next step (there are 2 columns like this), and then the data is ready for manipulation.  


### Manipulate Data
Final dataframe will have 87 columns = (43 data x 2) + subject_id (the "class"; as first column?)
