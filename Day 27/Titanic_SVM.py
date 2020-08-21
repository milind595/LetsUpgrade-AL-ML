
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


# In[2]:


data=pd.read_csv('train.csv')
data.head()


# In[3]:


data.info()


# In[4]:


data.shape


# In[5]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
data['Sex']=encoder.fit_transform(data['Sex'])
data['Embarked']=encoder.fit_transform(data['Embarked'])
data.head()


# In[6]:


data.isnull().sum()


# In[7]:


data=data.drop(['Cabin','PassengerId','Name','Ticket'],axis=1)
data.head()


# In[8]:


def Model(column):
    x = data.drop([column],axis=1)
    y = data[column]
    X_train,X_test,y_train,y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    SVM = SVC(gamma=0.01, C=100)
    SVM.fit(X_train, y_train)
    score = accuracy_score(SVM.predict(X_test),y_test, normalize=True)
    matrix = confusion_matrix(SVM.predict(X_test), y_test)
    print("The accuracy score is",score )
    print("The counfusion matrix is:\n",matrix)


# In[9]:


Model('Pclass')


# # Conclusion
# 
# Out of total 267 records only 241 are successfully classified, giving a model accuracy of 90%.

# In[10]:


Model('Sex')


# # Conclusion
# 
# Out of total 267 records only 194 are successfully classified, giving a model accuracy of 72.65%.

# In[11]:


Model('SibSp')


# # Conclusion
# 
# Out of total 267 records only 200 are successfully classified, giving a model accuracy of 75%.

# In[12]:


Model('Parch')


# # Conclusion
# 
# Out of total 267 records only 213 are successfully classified, giving a model accuracy of 80%.

# In[13]:


Model('Embarked')


# # Conclusion
# 
# Out of total 267 records only 205 are successfully classified, giving a model accuracy of 77%.
