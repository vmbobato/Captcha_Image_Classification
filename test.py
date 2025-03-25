import re

dataset_path = 'dataset_v1/'

# Remove all '/' and '\' characters using regex
cleaned_path = re.sub(r'[\\/]', '', dataset_path)

print(cleaned_path)