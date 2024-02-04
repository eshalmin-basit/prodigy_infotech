import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
import csv
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.title("The World Population Data Analytics Web App")

st.write(" ")
st.write(" ")
st.write(" ")

st.header("World Population Dataset")
data=pd.read_excel(r'D:\datascience_projects\prodigy infotech\task1\World_Bank_Data.xlsx')
st.dataframe(data)
data.head()
data_clean=data.dropna()

st.write(" ")
st.write("Removing null values")
st.dataframe(data_clean)
data_clean.head()
data_clean.isna().sum()
data_clean.info()
st.write(" ")
st.header("Population data filtered on the basis of continous variables")
st.write(" ")
data_fil=data_clean[['Birth rate, crude (per 1,000 people)',
                     'Death rate, crude (per 1,000 people)',
                     'Electric power consumption (kWh per capita)',
                     'GDP (USD)',
                     'GDP per capita (USD)',
                     'Individuals using the Internet (% of population)',
                     'Infant mortality rate (per 1,000 live births)',
                     'Life expectancy at birth (years)',
                     'Population density (people per sq. km of land area)',
                     'Unemployment (% of total labor force) (modeled ILO estimate)']]
df=pd.DataFrame(data_fil)
# Extract continuous variable columns
continuous_vars = df.select_dtypes(include='number').columns
u_year= sorted(data_clean['Year'].unique())
s_year= st.selectbox("Select a year",u_year)
s_data=data_clean[data_clean['Year'] == s_year]
st.header('Continuous Variable Distributions of Population Data for the selected year')
# Display histograms for each continuous variable
for var in continuous_vars:
    st.subheader(var)
    fig, ax = plt.subplots()
    ax.hist(s_data[var], bins=10, color='skyblue', edgecolor='black')
    ax.set_xlabel(var)
    ax.set_ylabel('Frequency')
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)
st.markdown("frequency in histograms provides a visual representation of the distribution of data, showing the concentration or spread of values within different ranges.")
st.markdown("It helps to identify patterns, central tendencies, and variability in the dataset.")


