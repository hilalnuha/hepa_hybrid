import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils import resample
# Import necessary libraries
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

file_path= "https://drive.google.com/file/d/1yQ7CYGYl8EOddaX7e1O0LP_JXlXAYy5p/view?usp=sharing"


# to get the id part of the file
id = file_path.split("/")[-2]

# downloaded_train = drive.CreateFile({'id':id})
# downloaded_train.GetContentFile('data_MISC_edit_CSV.csv')
!gdown --id $id -O data_hepa_modif_csv.csv

data = pd.read_csv("data_hepa_modif_csv.csv")
# Load the data
# data = pd.read_csv('/mnt/data/data_hepa_modif_csv.csv')

# Fill missing values
data_filled = data.copy()

# Fill numerical columns with median values
numerical_columns = data_filled.select_dtypes(include=['float64', 'int64']).columns
data_filled[numerical_columns] = data_filled[numerical_columns].fillna(data_filled[numerical_columns].median())

# Fill categorical columns with the mode (most frequent value)
categorical_columns = data_filled.select_dtypes(include=['object']).columns
data_filled[categorical_columns] = data_filled[categorical_columns].fillna(data_filled[categorical_columns].mode().iloc[0])

# Convert categorical variables to numerical using LabelEncoder
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    data_filled[column] = le.fit_transform(data_filled[column])
    label_encoders[column] = le

# Handle class imbalance: Upsample all classes
class_1 = data_filled[data_filled['Final Diagnosis'] == 1]
class_2 = data_filled[data_filled['Final Diagnosis'] == 2]
class_3 = data_filled[data_filled['Final Diagnosis'] == 3]

# Upsample the minority classes (class 1 and class 3)
class_1_upsampled = resample(class_1,
                             replace=True,     # sample with replacement
                             n_samples=len(class_2), # match majority class size
                             random_state=42) # set seed for reproducibility

class_3_upsampled = resample(class_3,
                             replace=True,     # sample with replacement
                             n_samples=len(class_2), # match majority class size
                             random_state=42) # set seed for reproducibility

# Combine the upsampled classes with class 2
data_balanced = pd.concat([class_1_upsampled, class_2, class_3_upsampled])

# Split the data into features (X) and target (y)
X = data_balanced.drop('Final Diagnosis', axis=1)
y = data_balanced['Final Diagnosis']

