#!/usr/bin/env python
# coding: utf-8

# ## Data Science Internship
# 
# ### Eshal Minhaj
# 
# ## Task 5: 
# Analyze traffic data to identify patterns related road conditions, weather and time of the day.
# 

# In[63]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os


# In[2]:


data=pd.read_csv(r'D:\datascience_projects\prodigy infotech\task5\us_acc.csv')
data.head()


# In[3]:


data.shape


# In[4]:


data.columns


# In[6]:


data.dtypes.value_counts()


# In[8]:


data.describe()


# In[10]:


data.State.unique()


# In[11]:


df1=data[data['State']=='CA']


# In[12]:


df1['IDD'] = df1['ID'].astype('str').str.extractall('(\d+)').unstack().fillna('').sum(axis=1).astype(int)


# In[13]:


df1


# In[14]:


df1.head()


# In[15]:


df1.shape


# In[17]:


df1.columns


# In[20]:


df1.duplicated().sum()


# In[21]:


d1f=df1.dropna(subset=['Precipitation(in)'])  


# In[22]:


df1.shape


# In[23]:


df1=df1.dropna(subset=['Temperature(F)','Wind_Chill(F)','Humidity(%)','Pressure(in)','Visibility(mi)','Wind_Direction', 'Wind_Speed(mph)',
                      'Weather_Condition'])


# In[24]:


df1.shape


# In[25]:


df1.isna().sum()/len(df1)*100


# In[26]:


df1=df1.dropna(subset=['City','Sunrise_Sunset',
       'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight'])


# In[27]:


df1.isna().sum()/len(df1)*100


# In[28]:


df1['Weather_Condition'].value_counts()


# In[32]:


df_cat=df1.select_dtypes('object')
df_num=df1.select_dtypes(np.number)
df_cat=df_cat.drop('ID',axis=1)


# In[33]:


df_cat=df1.select_dtypes('object')
col_name=[]
length=[]

for i in df_cat.columns:
    col_name.append(i)
    length.append(len(df_cat[i].unique()))
df_2=pd.DataFrame(zip(col_name,length),columns=['feature','count_of_unique_values'])
df_2


# num_col=df.select_dtypes('number') cat_col=df.select_dtypes('object') bool_col=df.select_dtypes('bool') float_col=df.select_dtypes('float64') int_col=df.select_dtypes('int64')

# In[34]:


df1.drop(['Description','Zipcode','Weather_Timestamp'],axis=1,inplace=True)


# In[35]:


del df1['Airport_Code']


# In[36]:


df_num.columns


# In[37]:


len(df_num.columns)


# In[38]:


df_cat.columns


# In[40]:


len(data['City'].unique())


# #### Numeric Data

# In[41]:


df_num=df1.select_dtypes(np.number)
col_name=[]
length=[]

for i in df_num.columns:
    col_name.append(i)
    length.append(len(df_num[i].unique()))
df_2=pd.DataFrame(zip(col_name,length),columns=['feature','count_of_unique_values'])
df_2


# In[42]:


plt.figure(figsize=(15 ,9))
sns.heatmap(df_num.corr() , annot=True)


# In[43]:


cities = df1['City'].unique()
len(cities)


# In[44]:


accidents_by_cities = df1['City'].value_counts()
accidents_by_cities


# In[45]:


#top 10 cities by number of accident
accidents_by_cities[:10]


# In[46]:


fig, ax = plt.subplots(figsize=(8,5))
accidents_by_cities[:10].plot(kind='bar')
ax.set(title = 'Top 10 cities By Number of Accidents',
       xlabel = 'Cities',
       ylabel = 'Accidents Count')
plt.show()


# In[47]:


accidents_severity = df1.groupby('Severity').count()['ID']
accidents_severity


# In[48]:


fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(aspect="equal"))
label = [1,2,3,4]
plt.pie(accidents_severity, labels=label,
        autopct='%1.1f%%', pctdistance=0.85)
circle = plt.Circle( (0,0), 0.5, color='white')
p=plt.gcf()
p.gca().add_artist(circle)
ax.set_title("Accident by Severity",fontdict={'fontsize': 16})
plt.tight_layout()
plt.show()


# In[49]:


df1['Start_Time'].dtypes


# In[50]:


df1['End_Time'].dtypes


# In[51]:


df1 = df1.astype({'Start_Time': 'datetime64[ns]', 'End_Time': 'datetime64[ns]'})
df1['Start_Time'].dtypes


# In[54]:


df1['Start_Time'][5041]


# In[55]:


df1['End_Time'][5041]


# In[56]:


df1['start_date'] = [d.date() for d in df1['Start_Time']]
df1['start_time'] = [d.time() for d in df1['Start_Time']]


# In[57]:


df1['end_date'] = [d.date() for d in df1['End_Time']]
df1['end_time'] = [d.time() for d in df1['End_Time']]


# In[58]:


df1['end_time']


# In[59]:


fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(df1['Start_Time'].dt.hour, bins = 24)

plt.xlabel("Start Time")
plt.ylabel("Number of Occurence")
plt.title('Accidents Count By Time of Day')

plt.show()


# In[60]:


fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(df1['Start_Time'].dt.hour, bins = 24)

plt.xlabel("End_Time")
plt.ylabel("Number of Occurence")
plt.title('Accidents Count By Time of Day')

plt.show()


# In[61]:


del df1['Start_Time']
del df1['End_Time']


# In[62]:


df1.head()


# In[70]:


fig, ax = plt.subplots(figsize=(8,5))
data['Weather_Condition'].value_counts(ascending=False)[:20].plot(kind='bar')
ax.set(title = 'Weather Conditions at Time of Accident Occurence',
       xlabel = 'Weather',
       ylabel = 'Accidents Count')
plt.show()


# most accidents happened when the weather was 'fair'. Perhaps weather (bad weather) was not a big contributing factor to accidents.

# In[71]:


# Accidents by order of severity (1 being lowest, and 4 being highest)

df1.groupby('Severity').count()['IDD']


# In[72]:


df_num.plot(kind='scatter', y='Start_Lat', x='Severity')


# In[73]:


sns.jointplot(x=df_num.Start_Lat.values , y=df_num.Start_Lng.values,height=10)
plt.ylabel('Start lattitude', fontsize=12)
plt.xlabel('Start lattitude', fontsize=12)
plt.show()


# In[74]:


sns.jointplot(x=df_num.End_Lat.values , y=df_num.End_Lng.values,height=10)
plt.ylabel('end lattitude', fontsize=12)
plt.xlabel('end longitude', fontsize=12)
plt.show()


# Summary and Conclusion
# Insights:
# 
# - No Data from New York
# - The number of accidents per city decreases exponentially.
# - Less than 5% of cities have more than 1000 yearly accidents.
# - Over 1100 cities have reported just one accident(need to investigate).

# In[75]:


import folium
from folium.plugins import HeatMap


# In[76]:


lat, lon = data.Start_Lat[0], data.Start_Lng[0]
lat, lon


# In[77]:


for x in data[['Start_Lat', 'Start_Lng']].sample(100).iteritems():
  print(x[1])


# In[78]:


zip(list(data.Start_Lat), list(data.Start_Lng))


# In[81]:


sample_df = data.sample(int(0.001*len(data)))
lat_lon_pairs = zip(list(sample_df.Start_Lat), list(sample_df.Start_Lng))


# In[82]:


map = folium.Map()
HeatMap(lat_lon_pairs).add_to(map)
map


# In[ ]:




