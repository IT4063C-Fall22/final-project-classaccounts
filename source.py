#!/usr/bin/env python
# coding: utf-8

# # Finding Food Insecurity

# ## Topic
# 
# I am trying to address food insecurity in the United States and identify geographic areas where food programs like the Supplemental Nutrition Assistance Program (SNAP), Special Supplemental Nutrition Program for Women (WIC),  National School Lunch Program (NSLP), and Emergency Food Assistance Program (TEFAP) could be targeted. Those organizations allocate funding based on geographic areas with the greatest need. It's essential to address this issue to ensure the right communities are getting the proper support through these programs to alleviate many health disparities caused by poor nutrition and food insecurity (FI).
# 
# 
# ## Project Question
# 
# Which counties of the united states are most impacted by food insecurity, and what indicators like business patterns, houselessness, and income are correlated to it?*
# 
# ## What would an answer look like?
# 
# The essential deliverable would be a choropleth map (code example included) showing the food insecurity index (FII) rate of each country in the United States. I would also have supporting line charts showing the FII rates compared to other indicators like houselessness and business patterns (specifically grocery stores filtered by North American Industry Classification System (NAICS)). The great thing about relating my dataset using Federal Information Processing Standards (FIPS) is that I can incorporate more datasets to correlate them to the FII of each county. Therefore, I believe I could have multiple answers to my question depending on what indicator correlated to FII we want to look at through exploratory data analysis. This would allow me to create very concrete answers for my question. Eventually, I would like to project food insecurity based on these indicators if I have the skills. However, that may be out of reach, given my current technical knowledge. But I can undoubtedly relate those indicators and likely correlate them to my FII per county.
# 
# 
# ## Data Sources
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
# ## Prior Feedback
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

# In[423]:


#Imports needed for the notebook
import pandas as pd
import plotly.figure_factory as ff
import requests
import plotly.express as px
import plotly as plt
from urllib.request import urlopen
import json
import seaborn as sns


# ### Choropleth Map Example
# 
# This is part of what my answer would look like. Please note the dataset can map values based on FIPS codes. I would replace the values with FII by county FIPS

# In[424]:


#This example was provided by plotly.com
fips = ['06021', '06023', '06027',
        '06029', '06033', '06059',
        '06047', '06049', '06051',
        '06055', '06061']
values = range(len(fips))
fig = ff.create_choropleth(fips=fips, values=values)
fig.layout.template = None
fig.show()


# ### Dataset #1 - County Business Patterns
# 
# This dataset is provided by the United States Census.
# 
# * Source Type: CSV (File)
# * Dataset URL: https://www.census.gov/programs-surveys/cbp/data/datasets.html
# * Documentation URL: https://www2.census.gov/programs-surveys/cbp/technical-documentation/records-layouts/2020_record_layouts/county-layout-2020.txt

# In[425]:


cbp = pd.read_csv("./datasources/cbp20co.txt")
cbp.sample(10)


# ### Dataset #1 - Feeding America Map the Meal Gap
# 
# This dataset is provided by the United States Census.
# 
# * Source Type: Excel (File)
# * Dataset URL: *You must create an account to access this*
# * Documentation URL: https://www.feedingamerica.org/research/map-the-meal-gap/overall-executive-summary

# In[426]:


#Dataset 2, Source: (File)
# https://www.feedingamerica.org/research/map-the-meal-gap/overall-executive-summary
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



# ### Dataset #1 - Census Population Survey
# 
# This dataset is provided by the United States Census.
# 
# * Source Type: API
# * Dataset URL: https://api.census.gov/data/2022/cps/basic/apr
# * Documentation URL: https://www.census.gov/data/developers/guidance/api-user-guide.html

# In[427]:


#Credit to Yahya Gilany course notes

HOST = "https://api.census.gov/data"
year = "2022"
dataset = "cps/basic/apr"
base_url = "/".join([HOST, year, dataset]) 
dataset_variables = ["GESTFIPS", "GTCO", "HEFAMINC"] 
predicates = {}
predicates["get"] = ",".join(dataset_variables) 
response = requests.get(base_url, params=predicates)
census_data = pd.DataFrame.from_records(response.json()[1:], columns=response.json()[0])
census_data.head()


# In[428]:


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
bls_df_18.head()


# ## Exploratory Data Analysis
# I will be exploring the MMG dataset first

# In[429]:


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


# In[430]:


#Rename columns
mmg_df_20_and_19.columns = ['fips', 'year', 'fi_rate', 'fi_pop', 'cost_per_meal']

#need to reorder mmg_df_20_and_19 since it is out of order
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
mmg_df_20_and_19.head()


# In[431]:


#add the year column,
mmg_df_18['year'] = 2018
mmg_df_17['year'] = 2017
mmg_df_16['year'] = 2016
mmg_df_15['year'] = 2015
mmg_df_14['year'] = 2014
mmg_df_13['year'] = 2013
mmg_df_12['year'] = 2012
mmg_df_11['year'] = 2011
mmg_df_10['year'] = 2010
mmg_df_18.head()


# In[457]:


#Merge the frames together to form one dataset, convert column to string
mmg_df = pd.concat([mmg_df_20_and_19, mmg_df_18, mmg_df_17, mmg_df_16, mmg_df_15, mmg_df_14, mmg_df_13, mmg_df_12, mmg_df_11, mmg_df_10])
mmg_df['year'] = mmg_df['year'].astype(float)
mmg_df['fips'] = mmg_df.fips.str.zfill(5)
mmg_df.head(20)


# In[433]:


#Merge datasets since they have similar columns
bls_df = pd.concat([bls_df_10, bls_df_11, bls_df_12, bls_df_13, bls_df_14, bls_df_15, bls_df_16, bls_df_17, bls_df_18, bls_df_19, bls_df_20])
bls_df.sample(20)


# In[434]:


#Drop unneeded columns, merge fips codes, and rename, scale down value of 
bls_df.drop(bls_df.columns[[0,3,5,6,7,8]], axis=1, inplace=True)
bls_df.columns = ['state_fips', 'county_fips', 'year', 'unemp_rate']
bls_df['fips'] = bls_df['state_fips'] + bls_df['county_fips']
bls_df.drop(bls_df.columns[[0,1]], axis=1, inplace=True)
bls_df['year'] = bls_df['year'].astype(float)
bls_df = bls_df[bls_df.unemp_rate != 'N.A.']
bls_df['unemp_rate'] = bls_df['unemp_rate'].astype(float)
bls_df['unemp_rate'] = bls_df['unemp_rate'].div(100)
bls_df.head()


# In[435]:


#master_df = pd.merge(, mmg_df, on={'year', 'fips'})
master_df = mmg_df.merge(bls_df, how='outer', on=['fips', 'year'])
master_df.sample(50)



# 

# In[ ]:


fig = px.line(
    data_frame=mmg_df.groupby(['year']).mean().reset_index(), 
    x="year", 
    y=['fi_rate']
)

fig.show()


# In[438]:


dfx = master_df.groupby(['year']).mean().reset_index().dropna()
fig = px.line(
    data_frame=dfx, 
    x="year", 
    y=[(dfx['unemp_rate'] - min(dfx['unemp_rate'])) / (max(dfx['unemp_rate']) - min(dfx['unemp_rate'])),
    (dfx['fi_rate'] - min(dfx['fi_rate'])) / (max(dfx['fi_rate']) - min(dfx['fi_rate']))]
)

fig.show()


# In[456]:


#This example was provided by plotly.com
mdf = mmg_df[mmg_df.year == 2010].dropna()
fig = ff.create_choropleth(fips=mdf['fips'], values=mdf['fi_rate'])
fig.layout.template = None
fig.show()


# In[ ]:


get_ipython().system('jupyter nbconvert --to python source.ipynb')

