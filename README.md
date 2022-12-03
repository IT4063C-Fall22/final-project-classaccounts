
# Finding Food Insecurity

## Topic

I am trying to address food insecurity in the United States and identify geographic areas where food programs like the Supplemental Nutrition Assistance Program (SNAP), Special Supplemental Nutrition Program for Women (WIC),  National School Lunch Program (NSLP), and Emergency Food Assistance Program (TEFAP) could be targeted. Those organizations allocate funding based on geographic areas with the greatest need. It's essential to address this issue to ensure the right communities are getting the proper support through these programs to alleviate many health disparities caused by poor nutrition and food insecurity (FI).


## Project Question

Which counties of the united states are most impacted by food insecurity, and what indicators like business patterns, houselessness, and income are correlated to it?*

## What would an answer look like?

The essential deliverable would be a choropleth map showing the food insecurity index (FII) rate of each country in the United States. I would also have supporting line charts showing the FII rates compared to other indicators like houselessness and business patterns (specifically grocery stores filtered by North American Industry Classification System (NAICS)). The great thing about relating my dataset using Federal Information Processing Standards (FIPS) is that I can incorporate more datasets to correlate them to the FII of each county. Therefore, I believe I could have multiple answers to my question depending on what indicator correlated to FII we want to look at through exploratory data analysis. Eventually, I would like to project food insecurity based on these indicators if I have the skills. However, that may be out of reach, given my current technical knowledge. But I can undoubtedly relate those indicators and likely correlate them to my FII per county.


## Data Sources

I have identified three datasets for this project. I will incorporate more as this project progresses and relate them using FIPS codes.

* Map the Meal Gap (MMG)
* County Business Patterns (CBP)
* Current Population Survey (CPS)

These datasets can be related using the FIPS code, which can indicate the county each row pertains to. All of these datasets I've imported contain a FIPS code. Therefore, I can join each of them using the county segment of the FIPS code.

I would be able to use the MMG dataset to identify the FII of each county. I can slice those visuals using the CPS dataset if I need to create reports based on a given demographic. When looking at correlated indicators related to FII, I will start with the CBP filtered dataset by grocery store NAICS codes and compare it to the MMG dataset joined on FIPS. I should be able to see trends of FII compared to trends in grocery store business spending and location patterns. I could also do this for other Census Housing/Income datasets and compare homelessness or income trends to FII by county or a higher level summary statistic of the country. I will also ensure I have to correct matching years for each dataset that they provide. However, only three datasets are required for this assignment at this time.

Please see more information regarding the datasets on the Jupyter notebook.

## Results

I was able to successfully create a Machine Learning model that can predict food insecurity rates for a given county in the United States. I will demonstrate this with the following choropleth maps. To see how I reached these results, please run my notebook and walk through each step from data preparation/cleaning, EDA, to ML modeling.

Please be sure to check out the interactive maps on the notebook, so you can view specific county food insecurity rates.

### 2020 Actual Food Insecurity Map
![alt text](https://github.com/IT4063C-Fall22/final-project-classaccounts/blob/main/images/2020_actual.png?raw=true)
### 2020 Machine Learning Model Predicted Food Insecurity Map
![alt text](https://github.com/IT4063C-Fall22/final-project-classaccounts/blob/main/images/2020_prediction.png?raw=true)
### 2020 Machine Learning Model Predicting Food Insecurity Map Without 2020 Training Data
Since the previous map was predicting 2020 actuals based on trained data from 2010 - 2020, I also wanted to create another trained model (2010 - 2019) that was not fitted with 2020 data and see how it performs predicting 2020 food insecurity, you can see the results below.
![alt text](https://github.com/IT4063C-Fall22/final-project-classaccounts/blob/main/images/2020_prediction_no2020train.png?raw=true)

## Installing Dependencies
- run `pipenv install`.
