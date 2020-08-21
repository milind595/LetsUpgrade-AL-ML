
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import accuracy_score,confusion_matrix


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


def metrics_model(a,b):
    x_train,x_test,y_train,y_test=train_test_split(a,b,test_size=0.3,random_state=0)
    s=(x_test.shape[0])+1
    for k in range(1,s):
        knn=neighbors.KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train,y_train).score(x_test,y_test)
        y_pred=knn.predict(x_test)
        print("The performance metric for the model with k =",k,"is:")
        print("The accuracy score is:",accuracy_score(y_test,y_pred,normalize=True))
        print("The confusion matrix is:")
        print(confusion_matrix(y_test,y_pred))


# In[9]:


y=data.iloc[:,1:2]
y.head()


# In[10]:


x=data.drop(["Pclass"],axis=1)
x.head()


# In[11]:


metrics_model(x,y)


# # Conclusion
# 
# The maximunm accuracy score is 89.13% for k = 1. 
