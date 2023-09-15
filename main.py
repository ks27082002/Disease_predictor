#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv('../DATA/heart.csv')


# In[4]:


df['target'].unique()


# In[5]:


df.info()


# In[6]:


df.describe().transpose()


# In[7]:


sns.countplot(x='target',data=df)


# In[8]:


df.columns


# In[9]:


sns.pairplot(df[['age','trestbps', 'chol','thalach','target']],hue='target')


# In[11]:


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),cmap='viridis',annot=True)


# In[12]:


X = df.drop('target',axis=1)
y = df['target']


# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)


# In[15]:


scaler = StandardScaler()


# In[16]:


scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)


# In[17]:


from sklearn.linear_model import LogisticRegressionCV 


# In[18]:


log_model = LogisticRegressionCV()


# In[19]:


log_model.fit(scaled_X_train,y_train)


# In[20]:


log_model.C_


# In[21]:


log_model.get_params()


# In[22]:


log_model.coef_


# In[23]:


coefs = pd.Series(index=X.columns,data=log_model.coef_[0])


# In[24]:


coefs = coefs.sort_values()


# In[25]:


plt.figure(figsize=(10,6))
sns.barplot(x=coefs.index,y=coefs.values);


# ---------
# 
# ## Model Performance Evaluation

# In[33]:


from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay


# In[34]:


y_pred = log_model.predict(scaled_X_test)


# In[35]:


confusion_matrix(y_test,y_pred)


# In[36]:


ConfusionMatrixDisplay.from_estimator(log_model,scaled_X_test,y_test)


# In[44]:


print(classification_report(y_test,y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[68]:


patient = [[ 54. ,   1. ,   0. , 122. , 286. ,   0. ,   0. , 116. ,   1. ,
          3.2,   1. ,   2. ,   2. ]]


# In[69]:


X_test.iloc[-1]


# In[70]:


y_test.iloc[-1]


# In[71]:


log_model.predict(patient)


# In[72]:


log_model.predict_proba(patient)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




