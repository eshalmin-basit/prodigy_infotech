#!/usr/bin/env python
# coding: utf-8

# # DATA SCIENCE INTERNSHIP
# 
# ## -ESHAL MINHAJ
# 
# ## TASK 2:
# ### DATA CLEANING AND EDA; Exploring Relationships, Patterns, & Trends in data.
# 
# Objective: To clean and explore the Titanic dataset, investigating relationships between variables, and uncovering patterns and trends to gain insights into factors influencing passenger survival on the Titanic.

# In[33]:


#import the libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import warnings
import warnings
warnings.filterwarnings('ignore')


# ## Reading the dataset

# In[34]:


#reading the dataset using the pandas lib
data=pd.read_csv(r'D:\datascience_projects\fun python projects\titanic\train.csv')
#observing the variables in the dataset
data.head() #displays the top 5 observations of the data


# In[35]:


data.tail() #displays the last 5 observations of the data


# In[36]:


data.shape #displays the no. of rows and columns in the dataset


# In[37]:


data.info() 
#data type and information about data, including the number of records in each column, 
#data having null or not null, Data type, the memory usage of the dataset


# so through the info() we can see that our dataset the columns Age, Embarked and Cabin has null values.
# The numeric variables has dtype as int64 anf float64 while the categorial variables have the dtype as objects.

# In[38]:


data.nunique()


# In[39]:


data.isnull().sum() #used to get the number of missing records in each column


# In[40]:


(data.isnull().sum()/(len(data)))*100 #gives the percentage of missing values in each column


# In[41]:


data.describe()


# In[42]:


data.count()


# In[43]:


#descriptive statistics of the object dtype
data.describe(include=['object'])


# In[44]:


#descriptive statistics of number dtype
data.describe(include=['number'])


# ### univariate and multivariate analysis using graphical and non graphical(some numbers represting the data)

# In[45]:


data.Survived.value_counts(normalize=True)


# as we can see around 38% of the passengers on the titanic survived while 61% couldn't make it

# ### Univariate analysis

# In[59]:


fig, axes= plt.subplots(2, 4, figsize=(16,10))
sns.countplot('Survived', data=data,ax=axes[0,0])
sns.countplot('Pclass', data=data, ax=axes[0,1])
sns.countplot('Sex',data=data, ax=axes[0,2])
sns.countplot('SibSp',data=data,ax=axes[0,3])
sns.countplot('Parch',data=data, ax=axes[1,0])
sns.countplot('Embarked',data=data, ax=axes[1,1])
sns.histplot(data['Fare'],kde=True, ax=axes[1,2])
sns.histplot(data['Age'].dropna(),kde=True, ax=axes[1,3])

for ax in axes.flatten():
    ax.grid(True, linestyle='--', alpha=0.7)


# 
# #### Countplot:
# The countplot is a categorical plot provided by the Seaborn library in Python. It is designed to show the counts of observations in each categorical bin using bars. This type of plot is particularly useful when you want to visualize the distribution of a categorical variable.
# 
# #### Axes variable :
# The axes variable will be a 2D array of Axes objects, where each element in the array corresponds to one subplot.
# When we create multiple subplots like this, we can then refer to each subplot using the axes array. For example, axes[0, 0] refers to the top-left subplot, and axes[1, 3] refers to the bottom-right subplot in the grid.
# 
# #### Histplot:
# Histplot is a function in the Seaborn library used for creating histograms. It is designed to visualize the distribution of a univariate set of observations. In addition to the classic histogram, histplot can include a kernel density estimate (KDE) for a smooth representation of the underlying distribution.
# 

# In[60]:


survival_rates = data.groupby('Sex')['Survived'].mean() * 100
print(survival_rates)
#comforming the observation


# #### Observations:
# 1. We can say that the male survival rate is around 20% while the female survival rate is around 75%, hence more females were able to survive the disaster and hence it can be said the survival rate has a close relation with the sex.
#     
# 2. Pclass 1 has a better survival rate around 60% followed by pclass 2 with 47% survival chance and then pclass 3 with the   survival rate of only 24%, hence the worst survival rate.
#     
# 3. Parch(passengers with parents or childern) =3 has a higher survival rate with 60% while parch=5 has a lower survival rate 
# 20% 
#     
# 4. There exsist a marginal relationship between fare and survival as from the obsevation we can say higher the fare the  
# chances of survival also increases. Also pclass=1 paid higher fare and that class has a higher chance of survival.
#     
# 5. There is an interesting point to note that the fare has is dependent on the age but is indirectly dependent on the class 
# for which the tickets are being purchased. So we can say that even if an individual with the age of 10 having pclass=3  
# could be paying lesser fare as compared to the individual in the same age group having the pclass=1

# In[63]:


survival_percentages_by_class = data.groupby('Pclass')['Survived'].mean() * 100
print(survival_percentages_by_class)


# In[66]:


survival_percentages_by_parch = data.groupby('Parch')['Survived'].mean() * 100
print(survival_percentages_by_parch)


# In[72]:


data['FamilySize'] = data['Parch'] + data['SibSp']

# Displaying the new 'FamilySize' column
print(data[['Parch', 'SibSp', 'FamilySize']])


# In[73]:


survival_percentages_by_f = data.groupby('FamilySize')['Survived'].mean() * 100
print(survival_percentages_by_f)


# In[87]:


survival_percentages_by_fare = data.groupby('Survived')['Fare'].mean() * 100
print(survival_percentages_by_fare)


# In[93]:


figbi, axesbi = plt.subplots(3, 4, figsize=(16, 10))
data.groupby('Pclass')['Survived'].mean().plot(kind='barh',ax=axesbi[0,0],xlim=[0,1])
data.groupby('SibSp')['Survived'].mean().plot(kind='barh',ax=axesbi[0,1],xlim=[0,1])
data.groupby('Parch')['Survived'].mean().plot(kind='barh',ax=axesbi[0,2],xlim=[0,1])
data.groupby('Sex')['Survived'].mean().plot(kind='barh',ax=axesbi[0,3],xlim=[0,1])
data.groupby('Embarked')['Survived'].mean().plot(kind='barh',ax=axesbi[1,0],xlim=[0,1])
data.groupby('FamilySize')['Survived'].mean().plot(kind='barh',ax=axesbi[1,3],xlim=[0,1])
sns.boxplot(x="Pclass",y="Fare",data=data,ax=axesbi[2,0])
sns.boxplot(x="Survived", y="Age", data=data,ax=axesbi[1,1])
sns.boxplot(x="Survived", y="Fare", data=data,ax=axesbi[1,2])
sns.boxplot(x="Pclass",y="Age",data=data,ax=axesbi[2,1])


# In[92]:


sns.jointplot(x="Age", y="Fare", data=data,ax=axesbi[2,1])


# ### MULTIVARIATE EDA

# In[97]:


f, ax=plt.subplots(figsize=(10,8))
corr=data.corr()
sns.heatmap(corr,
           mask=np.zeros_like(corr,dtype=bool),
           cmap=sns.diverging_palette(220,10,as_cmap=True),
           square=True,
           ax=ax)


# ##### Understanding a heatmap
# In a correlation matrix heatmap, features with higher correlation values are represented by brighter or darker colors. Positive correlations are often represented by shades of one color (e.g., dark blue), and negative correlations by shades of another color (e.g., dark red).
# 
# ##### Negetive Correlation:
# Negative correlation signifies that as one variable increases, the other variable tends to decrease. In other words, there is an inverse relationship between the two variables. When one variable goes up, the other tends to go down, and vice versa.
# For eg. when the pclass increases we can see the fare going down and vise versa.
# 
# ##### Positive Correlation:
# Positive correlation signifies that as one variable increases, the other variable tends to increase as well. In other words, there is a direct relationship between the two variables. When one variable goes up, the other tends to go up, and vice versa.
# 
# ##### Observations:
# 1. There is a positive relation with fare and survived 
# 2. There is a negetive correlation with age & pclass, fare & pclass, pclass & survived

# In[ ]:




