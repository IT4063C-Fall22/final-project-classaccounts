#!/usr/bin/env python
# coding: utf-8

# # Project Title
# Finding Food Insecurity

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 
# I am trying to address food insecurity in the united states and identify geographic areas where food programs like SNAP, WIC, NSLP, and TEFAP could be targeted. Those organizations typically allocate resources based on areas with the greatest need. It's important to address this issue to make sure the right communites are getting the right funding through these programs to alleviate many health disparities that are caused by poor nutrition and food insecurity.
# 
# 
# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 
# Which areas of the united states are most impacted by food insecurity, and whate indicators are closely linked to it?
# 
# I will determine the indicators as a progress and learn more about the project. A few indicators I have below would be overlayed by each counties FI. For example, on a line plot the FI of a given country would be overlayed by the trends of a busisness given x NAICS code. Or on a line plot the FI of a given country would be overlayed by the trends of residential income.
# 
# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 
# See below an example of a chloropleth map. This would be the main deliverable. However, I would also deliver line graphs comparing other indicators to the FI rate of a given county over x years. 
# 
# 
# ## Data Sources
# *What 3 data sources have you identified for this project?*
# 
# I will be using the MMG, CBP, and CPS datasets. All of these are written below in the notebook as PoC.
# 
# *How are you going to relate these datasets?*
# 
# Since the most atomic part of my geographic granularity will be by county, I will use the FIPS code in each dataset to join them.
# 
# *How will you use this data to answer your project question?*
# 
# I will be able to provide geographic visuals sliced by food FII to create a geographic chlorepleth map with aiding line chart visuals of FI trends in relation to time and indication.

# In[2]:


#This example was provided from plotly.com

#This is part of the What would an answer look like? 


import plotly.figure_factory as ff
fips = ['06021', '06023', '06027',
        '06029', '06033', '06059',
        '06047', '06049', '06051',
        '06055', '06061']
values = range(len(fips))
fig = ff.create_choropleth(fips=fips, values=values)
fig.layout.template = None
fig.show()


# In[4]:


import pandas as pd
#Dataset #1, Source: (File)
#The source of this dataset is from the country business patterns, https://www.census.gov/programs-surveys/cbp/data/datasets.html,
# https://www2.census.gov/programs-surveys/cbp/technical-documentation/records-layouts/2020_record_layouts/county-layout-2020.txt

cbp = pd.read_csv("./datasources/cbp20co.txt")
print(cbp.head())


# In[5]:


#Dataset 2, Source: (File)
# https://www.feedingamerica.org/research/map-the-meal-gap/overall-executive-summary
import pandas as pd

mmg = pd.read_excel("./datasources/MMG2022_2020-2019Data_ToShare.xlsx", sheet_name="County")
print(mmg.head())
print(mmg.describe())


# In[5]:


import requests

#Dataset #3, Source: (API)
# JOIN the variables with a `/` separator https://api.census.gov/data/2022/cps/basic/apr
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
print(census_data.head())


# In[3]:


get_ipython().system('jupyter nbconvert --to python source.ipynb')

