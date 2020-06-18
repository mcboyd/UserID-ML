# Per-subject data is easy enough
# How to get imposter data for each model?
#  - each imposter is a random combination of 18 other users' data
#  - maybe combine data from all users into one large dataset
#  - - use random.choice to randomly select 18 values in range of users excluding current user


# For each file in /arff_files/phone/accel
#  - import all paper-cited data for activity "A" (walking), moving class column to be first column

# For each file in /arff_files/phone/gyro
#  - import all paper-cited data for activity "A", appending to end (columns) of previous data for same user


import pandas
import pytorch
import random as rand
import tensorflow as tf

# Use while debugging
tf.enable_eager_execution()


activity = 'A'  # Code for walking activity
column_names = []  # List of column names to make sorting and filtering of imported data easier
all_data = pandas.df()  # Pandas dataframe to hold all subject data for the 1 activity (87 columns = 43 data x 2 + subject id as "class")
stats = zeros(1,51)  # Holds stats from model testing

for data_file in path('/arff_files/phone/accel'):
	temp = pandas.read(data_file)
	all_data.insert(temp.Activity[activity].cols[subject_id,data(1-43)])
	# Extract the subject ID (which is the last column in the file)
	# And the 43 columns of data for reach row where activity='A'
	# Insert in "all_data" with subject id as first column, followed by data columns

for data_file in path('/arff_files/phone/gyro'):
	temp = pandas.read(data_file)
	all_data.appendColumns(temp.Activity[activity].cols[data(1-43)]).joinOn(subject_id)
	# Extract the 43 columns of data for reach row where activity='A'
	# Append columns to "all_data" rows by matching on subject id
	# E.g., w/subject_id 1600, row "1600,1,2,...,43" becomes row "1600,1,2,...,43,g1,g2,...,g43" 

# Now all walking data, for all subjects, across 2 phone sensors is in "all_data" as:
# [subject_id],[a1],[a2],...,[a43],[g1],[g2],...,[g43]; where: 
# "subject_id" = the class to identify
# [a#] = the acceleramator data for that row for that subject
# [g#] = the gyroscope data for that row for that subject
# 87 columns
# 18 rows per subject x 52 subjects = 934 rows

# Now train the models
for subject in range(0:51):
	subject_idx = subject * 18
	# Below from https://stackoverflow.com/questions/17412439/how-to-split-data-into-trainset-and-testset-randomly
	subject_data = all_data[subject_idx:subject_idx+18,:]  # Pull out the subjects data
	random.shuffle(subject_data)  # Shuffle the subject data rows
	subject_train = subject_data[:9]  # Grab half the data rows for training
	subject_test = subject_data[9:]  # Grab half the data rows for testing
	
	# Get imposter data that excludes the current subject
	imp_data = imposter(subject)
	imp_train = imp_data(0-26,:)  # Take first 27 rows of imposter data for training 
	imp_test = imp_data(27-53,:)  # And the rest of the imposter data for testing

	# Train model
	model = train(subject_train, imp_train)

	# Test model
	stat = test(model, subject_test, imp_test)

	# Save model to disk w/ filename related to the subject #
	model.save_to_disk(subject)

	# Save stats
	stats[subject] = stat

print(stats)


# Generate imposter data for training: 
# 18 other subjects are randomly selected (not including passed-in "subject_id")
# 30 seconds of data for the same activity are randomly chosen for each "other" subject
def imposter(subject_id):
	imp_list = zeros(1,18)
	imp_data = zeros((18*3),87)  # 18 subjects, 30 secs data/subject; each row=10 secs => 3 rows/subject
	for j in range(1:19):
		imp_list[j] = rand.choice([for i in range(1600:1651) if i not in imp_list and i != subject_id])
	n = 1
	for imp in imp_list:
		imp_range = range((imp*18):(imp*19))  # Want the list of row numbers representing this imposter's data
		random.shuffle(imp_range)  # Shuffle the list of row numbers
		imp_rows = imp_range[:3]  # Grab 3 row numbers representing 3 random rows of data for this imposter
		r = 0
		for row in imp_rows:
			imp_data[(n+r),:] = all_data.row(row)
			r += 1
		n += 1
	imp_data.Subject_Id = 0  # Set the subject id column for all of the imposter data to be 0; this is their "class"
	return imp_data

# Defines a function that returns (features, labels) as required for TF train() and evaluate()
def make_input_fn(subject, imposter, batch_size, n_epochs=None):
	def input_fn():
                data = pd.concat([subject, imposter])
		labels = data.pop()
		dataset = tf.data.Dataset.from_tensor_slices((data, labels))
		dataset = dataset.shuffle(len(subject.index))
		dataset = dataset.repeat(n_epochs)
		dataset = dataset.batch(batch_size)
		return dataset
  	return input_fn

def train(subject, imposter):
	feature_columns = []
	for feature_name in column_names:
		feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
	# Train using batch size of sqrt(N)
	batch = floor(sqrt(len(subject.index)))
	train_input_fn = make_input_fn(subject, imposter, batch)
	model = tf.estimator.BoostedTreesClassifier(feature_columns, n_batches_per_layer=batch)
	model.train(train_input_fn, max_steps=100)
	return model


def test(model, subject, imposter):
	# Evaluate using full testing set
	n_samples = len(subject.index)
	eval_input_fn = make_input_fn(subject, imposter, n_samples, n_epochs=1)
	stat = model.evaluate(eval_input_fn)
	return stat


