#!/usr/bin/env python
# coding: utf-8

# # Finding Food Insecurity

# ## Writeup & Datasets *REVISE BEFORE FINAL PROJECT*
# ### Topic
# 
# I am trying to address food insecurity in the United States and identify geographic areas where food programs like the Supplemental Nutrition Assistance Program (SNAP), Special Supplemental Nutrition Program for Women (WIC),  National School Lunch Program (NSLP), and Emergency Food Assistance Program (TEFAP) could be targeted. Those organizations allocate funding based on geographic areas with the greatest need. It's essential to address this issue to ensure the right communities are getting the proper support through these programs to alleviate many health disparities caused by poor nutrition and food insecurity (FI).
# 
# 
# ### Project Question
# 
# Which counties of the united states are most impacted by food insecurity, and what indicators like business patterns, houselessness, and income are correlated to it?*
# 
# ### What would an answer look like?
# 
# The essential deliverable would be a choropleth map (code example included) showing the food insecurity index (FII) rate of each country in the United States. I would also have supporting line charts showing the FII rates compared to other indicators like houselessness and business patterns (specifically grocery stores filtered by North American Industry Classification System (NAICS)). The great thing about relating my dataset using Federal Information Processing Standards (FIPS) is that I can incorporate more datasets to correlate them to the FII of each county. Therefore, I believe I could have multiple answers to my question depending on what indicator correlated to FII we want to look at through exploratory data analysis. This would allow me to create very concrete answers for my question. Eventually, I would like to project food insecurity based on these indicators if I have the skills. However, that may be out of reach, given my current technical knowledge. But I can undoubtedly relate those indicators and likely correlate them to my FII per county.
# 
# 
# ### Data Sources
# 
# I have identified three datasets for this project. I will incorporate more as this project progresses and relate them using FIPS codes.
# 
# * Map the Meal Gap (MMG)
# * County Business Patterns (CBP)
# * Current Population Survey (CPS)
# 
# These datasets can be related using the FIPS code, which can indicate the county each row pertains to. All of these datasets I've imported contain a FIPS code. Therefore, I can join each of them using the county segment of the FIPS code.
# 
# I would be able to use the MMG dataset to identify the FII of each county. I can slice those visuals using the CPS dataset if I need to create reports based on a given demographic. When looking at correlated indicators related to FII, I will start with the CBP filtered dataset by grocery store NAICS codes and compare it to the MMG dataset joined on FIPS. I should be able to see trends of FII compared to trends in grocery store business spending and location patterns. I could also do this for other Census Housing/Income datasets and compare homelessness or income trends to FII by county or a higher level summary statistic of the country. I will also ensure I have to correct matching years for each dataset that they provide. However, only three datasets are required for this assignment at this time.
# 
# ### Prior Feedback
# 
# * *Is there enough data points that you'd be able to narrow the scope by that much?*
# 
# I have modified my scope to the county level inside the United States, and I can narrow it to this level using FIPS codes.
# 
# * *it seems that the project's scope is limited to descriptive analysis; I recommend digging a bit deeper and exploring other analysis types that might be helpful for this project.*
# 
# With my datasets I can do descriptive, predictive, exploratory, and inferential analysis. However, I really want to do predictive analyisis to find which countys may experience high FI in the future, but currently do not have the data analysis skills to do so.
# 
# * *you mentioned that you'd like to use "conformed dimensions from different datasets related to food insecurity". Are you able to identify some of those? if not the dataset itself, then a scope of what kind of information would like to get.*
# 
# The datasets I have identified can be used to get the information I need within my defined scope.

# In[349]:


#Imports needed for the notebook
import pandas as pd
import plotly.figure_factory as ff
import requests
import plotly.express as px
import plotly as plt
from urllib.request import urlopen
import json
import seaborn as sns


# ### Dataset #1 - County Business Patterns
# 
# This dataset is provided by the United States Census.
# 
# * Source Type: CSV (File)
# * Dataset URL: https://www.census.gov/programs-surveys/cbp/data/datasets.html
# * Documentation URL: https://www2.census.gov/programs-surveys/cbp/technical-documentation/records-layouts/2020_record_layouts/county-layout-2020.txt

# In[350]:


#Not in use at this checkpoint
cbp = pd.read_csv("./datasources/cbp20co.txt")


# ### Dataset #2 - Feeding America Map the Meal Gap
# 
# This dataset is provided by the United States Census.
# 
# * Source Type: Excel (File)
# * Dataset URL: *You must create an account to access this*
# * Documentation URL: https://www.feedingamerica.org/research/map-the-meal-gap/overall-executive-summary

# In[351]:


mmg_df_20_and_19 = pd.read_excel("./datasources/MMG2022_2020-2019Data_ToShare.xlsx", sheet_name="County", header=1, converters={0: str})
mmg_df_18 = pd.read_excel("./datasources/MMG2020_2018Data_ToShare.xlsx", sheet_name="2018 County", header=1, converters={0: str})
mmg_df_17 = pd.read_excel("./datasources/MMG2019_2017Data_ToShare.xlsx", sheet_name="2017 County", header=1, converters={0: str})
mmg_df_16 = pd.read_excel("./datasources/MMG2018_2016Data_ToShare.xlsx", sheet_name="2016 County", header=1, converters={0: str})
mmg_df_15 = pd.read_excel("./datasources/MMG2017_2015Data_ToShare.xlsx", sheet_name="2015 County", header=1, converters={0: str})
mmg_df_14 = pd.read_excel("./datasources/MMG2016_2014Data_ToShare.xlsx", sheet_name="2014 County", header=1, converters={0: str})
mmg_df_13 = pd.read_excel("./datasources/MMG2015_2013Data_ToShare.xlsx", sheet_name="2013 County", header=1, converters={0: str})
mmg_df_12 = pd.read_excel("./datasources/MMG2014_2012Data_ToShare.xlsx", sheet_name="2012 County", header=1, converters={0: str})
mmg_df_11 = pd.read_excel("./datasources/MMG2013_2011Data_ToShare.xlsx", sheet_name="2011 County", header=1, converters={0: str})
mmg_df_10 = pd.read_excel("./datasources/MMG2012_2010Data_ToShare.xlsx", sheet_name="County", header=1, converters={0: str})


# ### Dataset #3 - Small Area Income and Poverty Estimates (SAIPE)
# 
# This dataset is provided by the United States Census.
# 
# * Source Type: API
# * Dataset URL: https://api.census.gov/data/timeseries/poverty/saipe?
# * Documentation URL: https://api.census.gov/data/timeseries/poverty/saipe/variables.html

# In[352]:


def get_api_data(year):
    call = "https://api.census.gov/data/timeseries/poverty/saipe?get=GEOID,NAME,SAEMHI_PT,SAEPOVALL_PT,SAEPOVRTALL_PT,YEAR&for=county:*&time=" + year
    response = requests.get(call)
    census_data = pd.DataFrame.from_records(response.json()[1:], columns=response.json()[0])
    return census_data

pov_df_10 = get_api_data("2010")
pov_df_11 = get_api_data("2011")
pov_df_12 = get_api_data("2012")
pov_df_13 = get_api_data("2013")
pov_df_14 = get_api_data("2014")
pov_df_15 = get_api_data("2015")
pov_df_16 = get_api_data("2016")
pov_df_17 = get_api_data("2017")
pov_df_18 = get_api_data("2018")
pov_df_19 = get_api_data("2019")
pov_df_20 = get_api_data("2020")


# ### Dataset #4 - Local Area Unemployment Statistics
# This dataset is provided by the United States Census.
# 
# * Source Type: Excel (File)
# * Dataset URL: https://www.bls.gov/lau/
# * Documentation URL: https://www.bls.gov/lau/

# In[353]:


#datset #4
bls_df_10 = pd.read_excel("./datasources/laucnty10.xlsx", sheet_name="laucnty10", header=5, converters={1: str, 2: str})
bls_df_11 = pd.read_excel("./datasources/laucnty11.xlsx", sheet_name="laucnty11", header=5, converters={1: str, 2: str})
bls_df_12 = pd.read_excel("./datasources/laucnty12.xlsx", sheet_name="laucnty12", header=5, converters={1: str, 2: str})
bls_df_13 = pd.read_excel("./datasources/laucnty13.xlsx", sheet_name="laucnty13", header=5, converters={1: str, 2: str})
bls_df_14 = pd.read_excel("./datasources/laucnty14.xlsx", sheet_name="laucnty14", header=5, converters={1: str, 2: str})
bls_df_15 = pd.read_excel("./datasources/laucnty15.xlsx", sheet_name="laucnty15", header=5, converters={1: str, 2: str})
bls_df_16 = pd.read_excel("./datasources/laucnty16.xlsx", sheet_name="laucnty16", header=5, converters={1: str, 2: str})
bls_df_17 = pd.read_excel("./datasources/laucnty17.xlsx", sheet_name="laucnty17", header=5, converters={1: str, 2: str})
bls_df_18 = pd.read_excel("./datasources/laucnty18.xlsx", sheet_name="laucnty18", header=5, converters={1: str, 2: str})
bls_df_19 = pd.read_excel("./datasources/laucnty19.xlsx", sheet_name="laucnty19", header=5, converters={1: str, 2: str})
bls_df_20 = pd.read_excel("./datasources/laucnty20.xlsx", sheet_name="laucnty20", header=5, converters={1: str, 2: str})


# ## Data Cleaning & Transformations
# 
# We have 30 dataframes containing a decade of data that need to be concatenated and merged into a single master dataframe for exploratory data analysis (EDA). 
# 
# ### MMG Datasets
# 
# First, we need to drop unneeded columns to reduce the dataframes (DF) size. Since the MMG DFs have different columns for each dataset, we will need to drop them individually. We would also need to rename them to something that is more human-readable. Once the column headers are the same, we will need to add the year to each dataframe since the year is identified in the dataset file name, and that information is not available on a row level. We need to add it to the rows because we would merge all the dataframes on FIPS and year. Next, we need to concatenate all the individual MMG DFs into one. Once that is complete, I change the year to a float value on the concatenated MMG DF for EDA calculation reasons. In addition, we need to add leading 0's to the FIPS code since they were not padded in the dataset and are required for joins with other datasets. Finally, we multiply the FI rate to make it a percentage. 

# In[354]:


#Drop unneeded columns
mmg_df_20_and_19.drop(mmg_df_20_and_19.columns[[1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22]], axis=1, inplace=True)
mmg_df_18.drop(mmg_df_18.columns[[1,2,5,6,7,8,9,10,11,12,13,14,15,17]], axis=1, inplace=True)
mmg_df_17.drop(mmg_df_17.columns[[1,2,5,6,7,8,9,10,11,12,13,14,15,17]], axis=1, inplace=True)
mmg_df_16.drop(mmg_df_16.columns[[1,2,5,6,7,8,9,10,11,12,13,14,15,17]], axis=1, inplace=True)
mmg_df_15.drop(mmg_df_15.columns[[1,2,5,6,7,8,9,10,11,12,13,14,15,17]], axis=1, inplace=True)
mmg_df_14.drop(mmg_df_14.columns[[1,2,5,6,7,8,9,10,11,12,13,14,15,17]], axis=1, inplace=True)
mmg_df_13.drop(mmg_df_13.columns[[1,2,5,6,7,8,9,10,11,12,13,14,15,17]], axis=1, inplace=True)
mmg_df_12.drop(mmg_df_12.columns[[1,2,5,6,7,8,9,10,11,12,13,14,15,17]], axis=1, inplace=True)
mmg_df_11.drop(mmg_df_11.columns[[1,2,5,6,7,8,9,10,11,12,13,14,15,17]], axis=1, inplace=True)
mmg_df_10.drop(mmg_df_10.columns[[1,2,5,6,7,8,9,10,11,12,13,14,15,17]], axis=1, inplace=True)
mmg_df_10.head()

#Rename columns
mmg_df_20_and_19.columns = ['fips', 'year', 'fi_rate', 'fi_pop', 'cost_per_meal']
mmg_df_20_and_19 = mmg_df_20_and_19[['fips', 'fi_rate', 'fi_pop', 'cost_per_meal', 'year']]
mmg_df_18.columns = ['fips', 'fi_rate', 'fi_pop', 'cost_per_meal']
mmg_df_17.columns = ['fips', 'fi_rate', 'fi_pop', 'cost_per_meal']
mmg_df_16.columns = ['fips', 'fi_rate', 'fi_pop', 'cost_per_meal']
mmg_df_15.columns = ['fips', 'fi_rate', 'fi_pop', 'cost_per_meal']
mmg_df_14.columns = ['fips', 'fi_rate', 'fi_pop', 'cost_per_meal']
mmg_df_13.columns = ['fips', 'fi_rate', 'fi_pop', 'cost_per_meal']
mmg_df_12.columns = ['fips', 'fi_rate', 'fi_pop', 'cost_per_meal']
mmg_df_11.columns = ['fips', 'fi_rate', 'fi_pop', 'cost_per_meal']
mmg_df_10.columns = ['fips', 'fi_rate', 'fi_pop', 'cost_per_meal']

#Add year to columns with value
mmg_df_18['year'] = 2018
mmg_df_17['year'] = 2017
mmg_df_16['year'] = 2016
mmg_df_15['year'] = 2015
mmg_df_14['year'] = 2014
mmg_df_13['year'] = 2013
mmg_df_12['year'] = 2012
mmg_df_11['year'] = 2011
mmg_df_10['year'] = 2010

#Concat the datasets
mmg_df = pd.concat([mmg_df_20_and_19, mmg_df_18, mmg_df_17, mmg_df_16, mmg_df_15, mmg_df_14, mmg_df_13, mmg_df_12, mmg_df_11, mmg_df_10])

#Pad fips code, change year datatype, convert fi_rate decimal to percent
mmg_df['year'] = mmg_df['year'].astype(float)
mmg_df['fips'] = mmg_df.fips.str.zfill(5)
mmg_df['fi_rate'] = mmg_df['fi_rate'].multiply(100)
mmg_df.sample(5)


# ### BLS Datasets 
# 
# We can start by merging all the BLS dataframes since their datasets contain the same column headers and names. Next, we need to drop the unneeded columns to reduce dataset size and remove information we do not need to analyze. We also need to rename the columns into something more readable in the bls_df dataframe. The FIPS codes in this dataset are split into state and county segments. We need to merge those FIPS codes to have one FIPS code, which is required to join other datasets. Once that is complete, we can drop the state and county FIPS columns since we have a single FIPS code column. Next, we must cast the year and unemployment rate as a float for EDA calculations. However, first, we need to remove nonnumerical values from the unemployment rate before we cast it to a float, or else it will throw an exception.

# In[355]:


#Concatenate frames, drop uneeded columns, rename the columns
bls_df = pd.concat([bls_df_10, bls_df_11, bls_df_12, bls_df_13, bls_df_14, bls_df_15, bls_df_16, bls_df_17, bls_df_18, bls_df_19, bls_df_20])
bls_df.drop(bls_df.columns[[0,3,5,6,7,8]], axis=1, inplace=True)
bls_df.columns = ['state_fips', 'county_fips', 'year', 'unemp_rate']

#Concatentate the stat and county fips codes to new column, then drop them
bls_df['fips'] = bls_df['state_fips'] + bls_df['county_fips']
bls_df.drop(bls_df.columns[[0,1]], axis=1, inplace=True)

#Cast columns to correct datatype and remove non numerical values
bls_df['year'] = bls_df['year'].astype(float)
bls_df = bls_df[bls_df.unemp_rate != 'N.A.']
bls_df['unemp_rate'] = bls_df['unemp_rate'].astype(float)
bls_df.head()


# ### SAIPE Poverty Datasets
# 
# Since all the datasets have the same column headers, we can concatenate them into a new dataframe and drop the columns we will not need. We will also need to rename them into something more readable. Finally, the API call we made returned all values under the string data type. Therefore, we need to convert the rest of the values other than FIPS to float.

# In[356]:


#Concat dataframes and drop uneeded columns, and rename
pov_df = pd.concat([pov_df_10, pov_df_11, pov_df_12, pov_df_13, pov_df_14, pov_df_15, pov_df_16, pov_df_17, pov_df_18, pov_df_19, pov_df_20])
pov_df.drop(pov_df.columns[[6,7,8]], axis=1, inplace=True)
pov_df.columns = ['fips', 'county_name', 'med_income', 'tot_pop_pov', 'pov_rate', 'year']

#Modify data types as float
pov_df['med_income'] = pov_df['med_income'].astype(float)
pov_df['tot_pop_pov'] = pov_df['tot_pop_pov'].astype(float)
pov_df['pov_rate'] = pov_df['pov_rate'].astype(float)
pov_df['year'] = pov_df['year'].astype(float)


# ### Merging Datasets
# 
# Now that we have concatenated all the dataframes, the next step is to merge them, joined on fips and year. This will allow us to view the unemployment rate, food insecurity rate, poverty rate, and other values on one row where FIPS and year are the same. The rows will be inner joined to remove any null values.

# In[389]:


#Merging the datasets
master_df = mmg_df.merge(bls_df, how='inner', on=['fips', 'year'])
master_df = master_df.merge(pov_df, how='inner', on=['fips', 'year'])


# Below is descriptive information about the master dataframe. There would be no outliers unless we wanted to find the average population of a county since there are large population centers in the datframe.

# In[358]:


master_df.describe() 


# Since we inner joined the datasets, there are no null values, as seen below. Therefore, we will not have to remove or fill any null values.

# In[387]:


master_df.isnull().sum()


# As validated below, there are also no duplicate rows based on the year and FIPS.

# In[383]:


master_df[master_df.duplicated(['fips', 'year']) == True]


# Below is an example of what a typical County looks like. In our example, we will use Hamilton county Ohio.

# In[388]:


master_df[master_df['fips'] == "39061"]


# ## Data Visualizations
# 
# I have created multiple visualizations from my master dataframe. I will need to clean (adding labels, legends, etc.) up for the final project, but they are an excellent start for finding correlations and general EDA.
# 
# Below is a correlation matrix of all the values in the master dataframe. Some of the values are closely correlated, like the unemployment rate and food insecurity rate. 

# In[404]:


sns.heatmap(master_df.iloc[:,:10].corr(), center=0);


# In addition, we can also see the variation of the data using a box plot below, and this can help us identify any major outliers.

# In[405]:


sns.boxplot(data=master_df.iloc[0:10])


# The chart below takes our values with the highest correlation found in the correlation matrix and plots them over time based on the mean of each value. The values have been scaled to the same magnitude—credit to Erica Forehand for this formula. 

# In[407]:


dfx = master_df.groupby(['year']).mean().reset_index()
fig = px.line(
    data_frame=dfx, 
    x="year", 
    y=[(dfx['unemp_rate'] - min(dfx['unemp_rate'])) / (max(dfx['unemp_rate']) - min(dfx['unemp_rate'])),
    (dfx['fi_rate'] - min(dfx['fi_rate'])) / (max(dfx['fi_rate']) - min(dfx['fi_rate'])),
    (dfx['cost_per_meal'] - min(dfx['cost_per_meal'])) / (max(dfx['cost_per_meal']) - min(dfx['cost_per_meal'])),
    (dfx['pov_rate'] - min(dfx['pov_rate'])) / (max(dfx['pov_rate']) - min(dfx['pov_rate'])),
    (dfx['med_income'] - min(dfx['med_income'])) / (max(dfx['med_income']) - min(dfx['med_income']))],
)
fig.show()


# In the line chart above, unemployment, poverty, and food insecurity rates are closely correlated. Notice the unemployment rate in blue spikes, which indicates that there could be a significant increase in food insecurity in the future. This spike was due to COVID.
# 
# Also, the median income and cost per meal are strongly correlated and have an inverse correlation to the unemployment rate, poverty rate, and food insecurity rate.

# In[408]:


mdf = master_df[master_df.year == 2013]
fig = ff.create_choropleth(fips=mdf['fips'], values=mdf['fi_rate'])
fig.layout.template = None
fig.show()


# Above, you can see a choropleth map for Food insecurity in 2013. This is a great way to visually represent the United States geography and show which areas are most impacted by food security. We can also create another choropleth map to compare food insecurity and unemployment by county in 2013 below.

# In[411]:


mdf = master_df[master_df.year == 2020]
fig = ff.create_choropleth(fips=mdf['fips'], values=mdf['unemp_rate'])
fig.layout.template = None
fig.show()


# ## Machine Learning Plan
# 
# * What types of machine learning will you use in your project?
# 
# I want to project food insecurity based on correlated values for a given county. I will use a regression model.
# 
# * What issues do you see in making that happen?
# 
# I don't know how to make a regression model, and there could be an issue with underfitting the data.
# 
# * What challenges will you potentially face?
# 
# I have not previously used ML models in python for regression, so there could be a sharp learning curve.
# 
# (Sorry, I ran out of steam writing this problem, I'm sure I will have a much better answer once I have some ML implementation experience through labs. However, I'm confident I can create a model if it's anything like TensorFlow)
# 
# 

# In[ ]:


get_ipython().system('jupyter nbconvert --to python source.ipynb')

