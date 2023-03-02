# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 23:05:58 2023

@author: sachin kumar
"""
#importing the dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#data collection and preprocessing
raw_mail_data=pd.read_csv("D:/ml/spam Mail Prediction with Python/mail_data.csv")
mail_data=raw_mail_data.where((pd.notnull(raw_mail_data)),'')

#label encoding
#label spammail->0 ; ham mail->1
mail_data.loc[mail_data['Category']=='spam','Category',]=0;
mail_data.loc[mail_data['Category']=='ham','Category',]=1;

#seperating the text and labels
X=mail_data['Message']
Y=mail_data['Category']

#splitting the data into training and test data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3)

#transfrom the text data to feature vector that can be used as input to logistic regression
feature_extraction=TfidfVectorizer(min_df=1,stop_words='english',lowercase='True')
X_train_features=feature_extraction.fit_transform(X_train)
X_test_features=feature_extraction.transform(X_test)

#convert Y_train and Y_test to integer
Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')

#training the data 
#Logistic Regressions
model=LogisticRegression()

#traning the model with training data
model.fit(X_train_features,Y_train)


#predicting the accuracy of the training data
prediction_on_training_data=model.predict(X_train_features)
accuracy_on_training_data=accuracy_score(Y_train,prediction_on_training_data)
print("Accuracy on Training Data:",accuracy_on_training_data)

#predicting the accuracy of the test data
prediction_on_test_data=model.predict(X_test_features)
accuracy_on_test_data=accuracy_score(Y_test,prediction_on_test_data)
print("Accuracy on Test Data:",accuracy_on_test_data)


#predictive system
input_mail=[input("Enter the mail:")]
input_data_features=feature_extraction.transform(input_mail)
prediction=model.predict(input_data_features)
if(prediction[0]==1):
    print("HAM Mail")
else:
    print("SPAM mail")
    
    







