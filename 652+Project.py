
# coding: utf-8

# In[43]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("E:/First Semester/BIA 652/mushroom-classification/mushrooms.csv")
df[:5]


# In[44]:

df.describe()


# In[45]:

del df["veil-type"]
df.isnull().sum()


# In[46]:

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
for col in df.columns:
    df[col] = encoder.fit_transform(df[col])
df[:5]


# In[72]:

plt.figure()
pd.Series(df['class']).value_counts().sort_index().plot(kind = 'bar')
plt.ylabel("Count")
plt.xlabel("class")
plt.title('Number of poisonous/edible mushrooms (0=edible, 1=poisonous)')
plt.show()


# In[80]:

import seaborn as sns
plt.figure(figsize=(15,8))
sns.heatmap(df.corr(),annot=True, cmap="seismic_r",linewidths=0.5)
plt.show()


# In[60]:

df.corr()


# In[82]:

from sklearn.model_selection import train_test_split
y = df['class']
X = df.loc[:,df.columns != 'class']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)
from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)

predicted_output = logistic_regression.predict(X_test)


# In[83]:

from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(y_test, logistic_regression.predict_proba(X_test)[:,1], pos_label=1)
plt.plot(fpr,tpr, color="blue")
print("Logistic Regression auc = ", metrics.auc(fpr, tpr))
plt.show()


# In[84]:

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predicted_output)

print ("Accuracy of Logistic Regression is %f" %accuracy)


# In[92]:

y = df['class']
X = df[["gill-size","gill-color","bruises","ring-type"]]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)
logistic_regression1 = LogisticRegression()
logistic_regression1.fit(X_train, y_train)

predicted_output1 = logistic_regression1.predict(X_test)


# In[93]:

fpr1, tpr1, thresholds = metrics.roc_curve(y_test, logistic_regression.predict_proba(X_test)[:,1], pos_label=1)
plt.plot(fpr1,tpr1, color="red")
print("Logistic Regression auc = ", metrics.auc(fpr1, tpr1))
plt.show()


# In[95]:

accuracy1 = accuracy_score(y_test, predicted_output1)

print ("Accuracy of Logistic Regression is %f" %accuracy1)


# In[ ]:




# In[ ]:



