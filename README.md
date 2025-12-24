# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 
1.Load the dataset, drop unnecessary columns, and encode categorical variables.
2.Define the features (X) and target variable (y). 
3.Split the data into training and testing sets. 
4.Train the logistic regression model, make predictions, and evaluate using accuracy and other 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: vignesh s
RegisterNumber:  25014344
*/
[4:18 pm, 24/12/2025] Athil: import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1.head()

data1=data1.drop(['sl_no','salary'],axis=1)
data1.isnull().sum()
data1.duplicated().sum()
data1

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:, : -1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)

print("Accuracy score:",accuracy)
print("\nConfusion matrix:\n",confusion)
print("\nClassification Report:\n",cr)

from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()
[4:19 pm, 24/12/2025] Athil: Import the required packages and print the present data.
Print the placement data and salary data.
Find the null and duplicate values.
Using logistic regression find the predicted values of accuracy , confusion matrices.
Display the results.

```

## Output:
<img width="1351" height="318" alt="Screenshot 2025-12-24 164430" src="https://github.com/user-attachments/assets/faeb6dd8-8f33-4288-bcd5-5244b0f38871" />

<img width="1295" height="327" alt="Screenshot 2025-12-24 164437" src="https://github.com/user-attachments/assets/0b1a9247-1302-4ca7-a6bf-238023ed781d" />

<img width="1314" height="568" alt="Screenshot 2025-12-24 164444" src="https://github.com/user-attachments/assets/7ac148fc-da71-45ff-8dfc-ebdcd2a6f9ea" />

<img width="1339" height="385" alt="Screenshot 2025-12-24 164459" src="https://github.com/user-attachments/assets/24330807-5699-4604-8f20-b687a9971491" />

<img width="1297" height="594" alt="Screenshot 2025-12-24 164511" src="https://github.com/user-attachments/assets/42ae084f-11f5-4781-b234-a7458a05b7e9" />

<img width="934" height="456" alt="Screenshot 2025-12-24 164517" src="https://github.com/user-attachments/assets/294a156d-5259-4bae-9b5a-3d53c23ac768" />

<img width="1345" height="696" alt="Screenshot 2025-12-24 164522" src="https://github.com/user-attachments/assets/b4faf9dd-260b-403f-b5d1-6b058a705a47" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
