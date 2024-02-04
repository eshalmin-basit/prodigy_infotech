#!/usr/bin/env python
# coding: utf-8

# # Data Science Internship
# ### -Eshal Minhaj
# ## Task 3: Building a Decision Tree Classifier for Customer Purchase Prediction.
# 
# ### Objective:
# To create a decision tree classifier that can predict whether a customer will purchase a product or service based on their demographic and behavioral data using the UCI Machine Learning Repository's Bank dataset.

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import seaborn as sns
from sklearn.tree import export_text
from sklearn.tree import export_graphviz
import graphviz


# In[5]:


data=pd.read_csv(r'D:\datascience_projects\prodigy infotech\task3\bank.csv')
data.head()


# In[6]:


data.tail()


# In[8]:


data.shape


# In[9]:


data.isnull().sum()


# In[14]:


data.columns


# In[15]:


data.size


# In[17]:


data.describe()


# In[18]:


data.info()


# In[19]:


# Define the feature columns and target column
features = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
target = 'deposit'


# In[20]:


data = pd.get_dummies(data, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'], drop_first=True)


# In[21]:


# Split the data into training and testing sets
X = data.drop(target, axis=1)
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[22]:


X_train


# In[23]:


X_test


# In[24]:


y_train


# In[25]:


y_test


# In[26]:


# Create a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)


# In[27]:


clf.fit(X_train, y_train)


# ## Make Predictions

# In[28]:


y_pred = clf.predict(X_test)


# ## Evaluate the model

# In[29]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[30]:


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)


# In[31]:


class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)


# In[32]:


confusion = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Oranges")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# ## Visualize the tree structure

# In[33]:


tree_structure = export_text(clf, feature_names=list(X.columns))
print("Decision Tree Structure:")
print(tree_structure)


# In[ ]:




