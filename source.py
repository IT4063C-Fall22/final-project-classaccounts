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
# * Which counties of the united states are most impacted by food insecurity, and what indicators like poverty , unemployment, and income are correlated to it?*
# * How can we predict food insecurity in the future?
# 
# ### What would an answer look like?
# 
# The essential deliverable would be a choropleth map (code example included) showing the food insecurity index (FII) rate of each country in the United States. I would also have supporting line charts showing the FII rates compared to other indicators like povety and unemployment. The great thing about relating my dataset using Federal Information Processing Standards (FIPS) is that I can incorporate more datasets to correlate them to the FII of each county. Therefore, I believe I could have multiple answers to my question depending on what indicator correlated to FII we want to look at through exploratory data analysis. This would allow me to create very concrete answers for my question. Eventually, I would like to project food insecurity based on these indicators. I can undoubtedly relate those indicators and likely correlate them to my FII per county.
# 
# ### Data Sources
# 
# I have identified and imported four datasets for this project. However, I decided to only merge three of them into my combined dataset.
# 
# * Map the Meal Gap (MMG)
# * County Business Patterns (CBP)
# * Local Area Unemployment Statistics
# * Small Area Income and Poverty Estimates (SAIPE)
# 
# These datasets can be related using the FIPS code, which can indicate the county each row pertains to. All of these datasets I've imported contain a FIPS code. Therefore, I can join each of them using the county segment of the FIPS code.
# 
# I will explore their relations in the EDA section of this notebook. However, I will need to clean and merge the datasets first.

# In[88]:


#Imports needed for the notebook
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import matplotlib.pyplot as plt
from urllib.request import urlopen
import json
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import pickle

np.random.seed(42)


# ### Dataset #1 - County Business Patterns
# 
# This dataset is provided by the United States Census.
# 
# * Source Type: CSV (File)
# * Dataset URL: https://www.census.gov/programs-surveys/cbp/data/datasets.html
# * Documentation URL: https://www2.census.gov/programs-surveys/cbp/technical-documentation/records-layouts/2020_record_layouts/county-layout-2020.txt

# In[49]:


#Not in use at this checkpoint
cbp = pd.read_csv("./datasources/cbp20co.txt")


# ### Dataset #2 - Feeding America Map the Meal Gap
# 
# This dataset is provided by Feeding America.
# 
# * Source Type: Excel (File)
# * Dataset URL: *You must create an account to access this*
# * Documentation URL: https://www.feedingamerica.org/research/map-the-meal-gap/overall-executive-summary

# In[50]:


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

# In[51]:


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

# In[52]:


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

# In[53]:


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

# In[54]:


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

# In[55]:


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

# In[56]:


#Merging the datasets
master_df = mmg_df.merge(bls_df, how='inner', on=['fips', 'year'])
master_df = master_df.merge(pov_df, how='inner', on=['fips', 'year'])


# Below is descriptive information about the master dataframe. There would be no outliers unless we wanted to find the average population of a county since there are large population centers in the datframe.

# In[57]:


master_df.describe() 


# Since we inner joined the datasets, there are no null values, as seen below. Therefore, we will not have to remove or fill any null values.

# In[58]:


master_df.isnull().sum()


# As validated below, there are also no duplicate rows based on the year and FIPS.

# In[59]:


master_df[master_df.duplicated(['fips', 'year']) == True]


# Below is an example of what a typical County looks like. In our example, we will use Hamilton county Ohio.

# In[60]:


master_df[master_df['fips'] == "39061"]


# ## Data Visualizations
# 
# I have created multiple visualizations from my master dataframe. I will need to clean (adding labels, legends, etc.) up for the final project, but they are an excellent start for finding correlations and general EDA.
# 
# Below is a correlation matrix of all the values in the master dataframe. Some of the values are closely correlated, like the unemployment rate and food insecurity rate. 

# In[89]:


ax = plt.axes()
sns.heatmap(master_df.corr(), center=0, ax=ax)
ax.set_title('Combined Dataset Correlation Matrix')


# In addition, we can also see the variation of the data using a box plot below, and this can help us identify any major outliers.

# In[90]:


ax = plt.axes()
sns.boxplot(master_df.iloc[0:3], ax=ax)
ax.set_title('Combine Dataset Variation')


# The chart below takes our values with the highest correlation found in the correlation matrix and plots them over time based on the mean of each value. The values have been scaled to the same magnitude—credit to Erica Forehand for this formula. 

# In[110]:


dfx = master_df.groupby(['year']).mean().reset_index()
fig = px.line(
    data_frame=dfx, 
    x="year", 
    y=[(dfx['unemp_rate'] - min(dfx['unemp_rate'])) / (max(dfx['unemp_rate']) - min(dfx['unemp_rate'])),
    (dfx['fi_rate'] - min(dfx['fi_rate'])) / (max(dfx['fi_rate']) - min(dfx['fi_rate'])),
    (dfx['cost_per_meal'] - min(dfx['cost_per_meal'])) / (max(dfx['cost_per_meal']) - min(dfx['cost_per_meal'])),
    (dfx['pov_rate'] - min(dfx['pov_rate'])) / (max(dfx['pov_rate']) - min(dfx['pov_rate'])),
    (dfx['med_income'] - min(dfx['med_income'])) / (max(dfx['med_income']) - min(dfx['med_income']))],
    title='Scaled Mean Value of Important Features over Time'
)
newnames = {'wide_variable_0':'Unemployment Rate', 'wide_variable_1':'Food Insecurity Rate', 'wide_variable_2':'Cost Per Meal', 'wide_variable_3':'Poverty Rate', 'wide_variable_4':'Median Income'}
#https://www.folkstalk.com/tech/plotly-change-legend-name-with-code-examples/ - it was very hard to change legend names when calculated inside of plot
fig.for_each_trace(lambda legend: 
legend.update(name = newnames[legend.name],
legendgroup=newnames[legend.name],
hovertemplate=legend.hovertemplate.replace(legend.name, newnames[legend.name])))
fig.show()


# In the line chart above, unemployment, poverty, and food insecurity rates are closely correlated. Notice the unemployment rate in blue spikes, which indicates that there could be a significant increase in food insecurity in the future. This spike was due to COVID. We would expct the food insecurity rat to also spike, but it may not due to support from governmental programs.
# 
# Also, the median income and cost per meal are strongly correlated and have an inverse correlation to the unemployment rate, poverty rate, and food insecurity rate.

# In[86]:


mdf = master_df[master_df.year == 2015]
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",
                   dtype={"fips": str})
fig = px.choropleth(mdf, geojson=counties, locations='fips', color='fi_rate',
                           color_continuous_scale="Jet",
                           range_color=(mdf.fi_rate.min(), mdf.fi_rate.max()),
                           scope="usa",
                           labels={'fi_rate':'2020 Food Insecurity Rate %'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[65]:


fig = px.choropleth(master_df[master_df.year == 2010], geojson=counties, locations='fips', color='fi_rate',
                           color_continuous_scale="Jet",
                           range_color=(mdf.fi_rate.min(), mdf.fi_rate.max()),
                           scope="usa",
                           labels={'fi_rate':'2010 Food Insecurity Rate %'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# Above, you can see a choropleth map for Food insecurity in 2013. This is a great way to visually represent the United States geography and show which areas are most impacted by food security. We can also create another choropleth map to compare food insecurity and unemployment by county in 2013 below.

# In[84]:


mdf = master_df[master_df.year == 2013]
fig = px.choropleth(mdf, geojson=counties, locations='fips', color='unemp_rate',
                           color_continuous_scale="Jet",
                           range_color=(0, mdf.unemp_rate.max()),
                           scope="usa",
                           labels={'unemp_rate':'2013 Unemployment Rate'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[109]:


fig = px.choropleth(mdf, geojson=counties, locations='fips', color='pov_rate',
                           color_continuous_scale="Jet",
                           range_color=(0, mdf.pov_rate.max()),
                           scope="usa",
                           labels={'pov_rate':'2013 Poverty Rate'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# ## Machine Learning & Predictive Model
# In this section I will build a predictive regression model to estimate the food insecurity rate of a given county

# Let's start by creating a copy of my merged dataframe below.

# In[68]:


fi_model = master_df.copy()


# The fi_model needs to be split into testing and training. I will not be using a stattified split since an equal split on some variable is not needed.
# 
# I will also drop the features of the fi_model (fi_X) that I do not need and create the target series (fi_rate, fi_y)
# 
# In addition, I'm dropping the FI population and cost_per_meal since it came from the target dataset and was derived from it.

# In[69]:


fi_train_set, fi_test_set = train_test_split(fi_model, random_state=42, test_size=0.2)
fi_X = fi_train_set.drop(['fi_rate', 'county_name', 'fips', 'fi_pop', 'cost_per_meal', 'year'], axis=1)
fi_y = fi_train_set['fi_rate'].copy()


# The model pipeline below contains the steps to fit and transform my fi_model. First, we will use a custom transformer to get some engineered features like the total population which is derived from tot_pop_pov and pov_rate. Next, we use as simple imputer and standard scaler to further improve the output of our prepared model. The standard scaler was very important for this project due to the range of data.

# In[70]:


num_features = fi_X.columns

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['tot_pop'] = X['tot_pop_pov'] / (X['pov_rate'] * 0.01)
        return X

num_pipeline = Pipeline([
    ('fe', FeatureEngineer()),
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_features)
])

fi_X_prepared = full_pipeline.fit_transform(fi_X)


# I tested multiple regressors and got the lowest RMSE score and best CV by using the Random Forest Regressor. In addition, I've attempted to tune it with some hyper parameters. This is the best model for predicting our target of food insecurity rate.

# In[71]:


rfr = Pipeline([
    ('lin', RandomForestRegressor(random_state=42, n_estimators=250))
])
rfr.fit(fi_X_prepared, fi_y)


# Finding the training set RMSE below.

# In[72]:


predictions = rfr.predict(fi_X_prepared)
rfr_mse = mean_squared_error(fi_y, predictions)
rfr_rmse = np.sqrt(rfr_mse)
print('The Train Random Forest Regressor RMSE:', rfr_rmse)


# Finding the predictive model Cross Validation score below.

# In[73]:


kfold = KFold(n_splits=10)
scores = cross_val_score(rfr, fi_X, fi_y, cv=kfold,  scoring='r2')
print('The Mean Random Forest Regressor CV Score:', abs(np.mean(scores)))


# Testing set RMSE below.

# In[74]:


fi_test_X = fi_test_set.drop(['fi_rate', 'county_name', 'fips', 'year', 'cost_per_meal'], axis=1)
fi_test_y = fi_test_set['fi_rate'].copy()
X_test_clean = full_pipeline.transform(fi_test_X)
predictions_poly = rfr.predict(X_test_clean)
rfr_mse = mean_squared_error(fi_test_y, predictions_poly)
test_rfr_rmse = np.sqrt(rfr_mse)
print('The Test Random Forest Regressor RMSE:', test_rfr_rmse)


# While the results for my model may not look great at first glance, the mean margin of error is typically off by only 0.8%, which is not bad. We can check this by comparing the actual food insecurity rate vs the predicted food insecurity rate. I will also demonstrate this in the Results section of this notebook.
# 
# In addition, I also compared the actuals of 2020 to the predictions of a similiar trained model where I remove data from the year 2020. The best model contains training data from 2010 - 2020 where the no2020 contains training data from 201-2019. I did this to test how well the regression model is working when the data we are trying to predict was not included in the training.

# In[75]:


fi_model_no2020  = master_df[master_df['year'] != 2020]
fi_train_set_no2020 , fi_test_set_no2020  = train_test_split(fi_model_no2020 , random_state=42, test_size=0.2)
fi_X_no2020  = fi_train_set_no2020 .drop(['fi_rate', 'county_name', 'fips', 'fi_pop', 'cost_per_meal', 'year'], axis=1)
fi_y_no2020  = fi_train_set_no2020 ['fi_rate'].copy()
num_features_no2020  = fi_X_no2020 .columns
class FeatureEngineer_no2020(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['tot_pop'] = X['tot_pop_pov'] / (X['pov_rate'] * 0.01)
        return X
num_pipeline_no2020  = Pipeline([
    ('fe', FeatureEngineer_no2020()),
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
full_pipeline_no2020  = ColumnTransformer([
    ('num', num_pipeline_no2020, num_features_no2020)
])
fi_X_prepared_no2020  = full_pipeline_no2020.fit_transform(fi_X_no2020)
rfr_no2020  = Pipeline([
    ('lin', RandomForestRegressor(random_state=42, n_estimators=250))
])
rfr_no2020.fit(fi_X_prepared_no2020, fi_y_no2020)


# The code below creates a table of the actual food insecurity rates for 2020 vs. the predicted food insecurity of 2020 using my best model vs. the predicted food insecurity rate of 2020 using a clone of my best model that was not trained on 2020 data.

# In[76]:


fi_2020 = master_df[master_df['year'] == 2020]
fi_2020_X = fi_2020.drop(['fi_rate', 'county_name', 'fips', 'fi_pop', 'cost_per_meal', 'year'], axis=1)
fi_2020_y = pd.DataFrame(fi_2020['fi_rate'].copy())
rfr_no2020.predict(full_pipeline_no2020.transform(fi_2020_X))
comp_df = pd.DataFrame({
    'fips': fi_2020['fips'].values,
    'fi_rate':fi_2020_y['fi_rate'].values,
    'pred_fi':rfr.predict(full_pipeline.transform(fi_2020_X)),
    'pred_fi_no2020':rfr_no2020.predict(full_pipeline_no2020.transform(fi_2020_X))
})
comp_df['pred_fi'] = comp_df['pred_fi'].round(1)
comp_df['pred_fi_no2020'] = comp_df['pred_fi_no2020'].round(1)
comp_df['margin_of_error_best_model'] = abs(comp_df['fi_rate'] - comp_df['pred_fi'])
comp_df['margin_of_error_best_model_no2020'] = abs(comp_df['fi_rate'] - comp_df['pred_fi_no2020'])
display(comp_df.sample(10))
print('The mean error for the best model is', round(comp_df.margin_of_error_best_model.mean(),1), 'for the dataset')
print('The mean error for the best model without 2020 training data is', round(comp_df.margin_of_error_best_model_no2020.mean(),1), 'for the dataset')


# ## Results
# 
# The best way to demonstrate the result of my final project is to create a choropleth map comparing the actual food insecurity rate to my Random Forest regression model. The first map shows the actual food insecurity rate for 2020 and the second map shows the predicted food insecurity rate for 2020 using my model.

# ### 2020 Actual Food Insecurity Map

# In[77]:


fig = px.choropleth(comp_df, geojson=counties, locations='fips', color='fi_rate',
                           color_continuous_scale="Jet",
                           range_color=(min(comp_df[['pred_fi', 'fi_rate', 'pred_fi_no2020']].min(axis=0)), max(comp_df[['pred_fi', 'fi_rate', 'pred_fi_no2020']].max(axis=0))),
                           scope="usa",
                           labels={'fi_rate':'2020 Food Insecurity Actuals'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# ### 2020 Machine Learning Model Predicted Food Insecurity Map

# In[78]:


fig = px.choropleth(comp_df, geojson=counties, locations='fips', color='pred_fi',
                           color_continuous_scale="Jet",
                           range_color=(min(comp_df[['pred_fi', 'fi_rate', 'pred_fi_no2020']].min(axis=0)), max(comp_df[['pred_fi', 'fi_rate', 'pred_fi_no2020']].max(axis=0))),
                           scope="usa",
                           labels={'pred_fi':'2020 Food Insecurity Predictions'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# ### 2020 Machine Learning Model Predicting Food Insecurity Map Without 2020 Training Data
# 
# Since the previous chart was predicting 2020 actuals based on trained data from 2010 - 2020, I also wanted to create another trained model (2010 - 2019) that was not fitted with 2020 data and see how it performs predicting 2020 food insecurity, you can see the results below.

# In[79]:


fig = px.choropleth(comp_df, geojson=counties, locations='fips', color='pred_fi_no2020',
                           color_continuous_scale="Jet",
                           range_color=(min(comp_df[['pred_fi', 'fi_rate', 'pred_fi_no2020']].min(axis=0)), max(comp_df[['pred_fi', 'fi_rate', 'pred_fi_no2020']].max(axis=0))),
                           scope="usa",
                           labels={'pred_fi_no2020':'2020 Food Insecurity Predictions Without 2020 Training'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# Below is an example of how you can predict a county's food insecurity at an ad hoc level. However, there is not enough poverty or unemployment data since 2020 to make these type of predictions

# In[80]:


pred = pd.DataFrame({
    'pov_rate': [16.9],
    'tot_pop_pov': [7426.0],
    'med_income': [51751.0],
    'unemp_rate': [5.1],
})
pred_trans = full_pipeline.transform(pred)
print(rfr.predict(pred_trans))


# Saving the best model and all features using pickle.
# 
# ***DO COMMIT THESE FILES, THEY ARE TOO LARGE FOR GITHUB.***

# In[81]:


#pickle.dump(master_df, file=open('all_features.pkl', 'wb'))
#pickle.dump(rfr, file=open('predictive_model.pkl', 'wb'))


# In[82]:


get_ipython().system('jupyter nbconvert --to python source.ipynb')

