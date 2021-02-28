# -*- coding: utf-8

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
# %matplotlib inline
from numpy import random
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score,classification_report ,confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
import joblib



my_data= pd.read_csv('healthcare-dataset-stroke-data.csv')
#my_data['Age'] =round(my_data['Age'], 1)
#my_data['Tenure'] =round(my_data['Tenure'], 1)
#my_data.describe()
#first few rows

my_data.head()

my_data.dropna()

my_data.fillna(0, inplace=True)

my_data_numeric =my_data[['age','hypertension','heart_disease','avg_glucose_level','bmi','stroke']]

my_data_category =  my_data[['gender','smoking_status']]

# Remove the categorical variable 
#my_data =  my_data.drop(['gender','smoking_status'], axis=1)

SMOKING_STATUS = pd.get_dummies(my_data_category.smoking_status).iloc[:,1:]
GENDER = pd.get_dummies(my_data_category.gender).iloc[:,1:]

my_data_model =my_data_numeric

my_data_model.head()

X =  my_data_model.drop(['stroke'], axis=1)
#X = my_data[['Tenure', 'Age','CHANNEL4_6M','CHANNEL4_3M','PAYMENTS_6M','METHOD1_6M','CHANNEL2_6M','PAYMENTS_3M','LOGINS']]
y = my_data_model['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


print(y_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=200, random_state=0)  
classifier.fit(X_train, y_train)  
predictions = classifier.predict(X_test)
predictions_prob = classifier.predict_proba(X_test)

from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test,predictions ))  
print(accuracy_score(y_test, predictions ))

feat_importances = pd.Series(classifier.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')

log_loss(y_test, predictions_prob)

# Saving model to disk
pickle.dump(classifier, open('model.pkl','wb'))



  
