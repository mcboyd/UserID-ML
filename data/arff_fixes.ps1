# Helpful code:
# https://stackoverflow.com/questions/14440062/how-to-remove-some-words-from-all-text-file-in-a-folder-by-powershell

# Set paths below as appropriate to local location of arff files

# PHONE ACCEL 
ls wisdm-dataset\arff_files\phone\accel\*.arff | %{ $newcontent=(gc $_) -replace '"','' -replace '{ ','{' -replace ' }','}' -replace ', ',',' |sc $_ }

# PHONE GYRO
ls wisdm-dataset\arff_files\phone\gyro\*.arff | %{ $newcontent=(gc $_) -replace '"','' -replace '{ ','{' -replace ' }','}' -replace ', ',',' |sc $_ }