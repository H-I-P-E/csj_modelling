from boto3 import client
from os import getenv
from io import StringIO
from decouple import config
import pandas as pd


#Connect to the data in s3
s3_client = client('s3',aws_access_key_id = config("ACCESS_KEY"), aws_secret_access_key = config("SECRET_KEY"))

sample_files = ["2022-03-10_192592_193511_.csv", "2022-03-09_183221_192425_.csv", "2022-03-09_189446_192862_.csv", "2022-03-09_190252_193075_.csv", "2022-03-10_193018_193333_.csv", "2022-03-10_193347_192802_.csv"]

#Load all the sample data files
all_raw_data = []
for file in sample_files:
    raw_data_file = s3_client.get_object(Bucket="civil-service-jobs", Key="raw_data/" + file)
    raw_data = pd.read_csv(StringIO(raw_data_file['Body'].read().decode('utf-8')))
    all_raw_data.append(raw_data)
combined_data = pd.concat(all_raw_data)

columns = ["date_downloaded", "department", "Job description", "title", "Grade"]
#combined_data =combined_data['variable'].str.contains('|'.join(columns),regex=True)

print(combined_data['variable'].value_counts())
