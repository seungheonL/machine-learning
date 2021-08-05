# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importing the dataset
dataset = pd.read_csv('/workspace/MachineLearning/data-preprocessing/Data.csv')
X = dataset.iloc[:, :-1].values # [a:b, c:d] = a행~b행, c행~d행 선택
y = dataset.iloc[:, -1].values

# Taking care of missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # np.nan인 값들을 평균값으로 대체
imputer.fit(X[:, 1:3]) # 행 전체, 1열~3열에 대하여 평균 계산
X[:, 1:3] = imputer.transform(X[:, 1:3]) # 적용

# Encoding categorical data

# Encoding the Independent Variable
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough') # 각 국가들을 인코딩함
X = np.array(ct.fit_transform(X))

# Encoding the Dependent Variable
le = LabelEncoder() # Yes, No를 인코딩함
y = le.fit_transform(y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 20% of test set, 80% of training set

# Feature Scaling
sc = StandardScaler() # 각 수치들을 표준화함
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:]) 
X_test[:, 3:] = sc.fit_transform(X_test[:, 3:])
