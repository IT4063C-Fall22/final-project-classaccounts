
# Finding Food Insecurity

## Installing Dependencies

- run `pipenv install`.

## Topic

I am trying to address food insecurity in the United States and identify geographic areas where food programs like the Supplemental Nutrition Assistance Program (SNAP), Special Supplemental Nutrition Program for Women (WIC),  National School Lunch Program (NSLP), and Emergency Food Assistance Program (TEFAP) could be targeted. Those organizations allocate funding based on geographic areas with the greatest need. It's essential to address this issue to ensure the right communities are getting the proper support through these programs to alleviate many health disparities caused by poor nutrition and food insecurity (FI).

## Project Question

* Which counties of the united states are most impacted by food insecurity, and what indicators like business patterns, houselessness, and income are correlated to it?
* How can we predict food insecurity in the future?

## What would an answer look like?

The essential deliverable would be a choropleth map (code example included) showing the food insecurity index (FII) rate of each country in the United States. I would also have supporting line charts showing the FII rates compared to other indicators like povety and unemployment. The great thing about relating my dataset using Federal Information Processing Standards (FIPS) is that I can incorporate more datasets to correlate them to the FII of each county. Therefore, I believe I could have multiple answers to my question depending on what indicator correlated to FII we want to look at through exploratory data analysis. This would allow me to create very concrete answers for my question. Eventually, I would like to predict food insecurity based on these indicators. I can undoubtedly relate those indicators and likely correlate them to my FII per county.

## Data Sources

I have identified and imported four datasets for this project. However, I decided to only merge three of them into my combined dataset.

* Map the Meal Gap (MMG)
* County Business Patterns (CBP)
* Local Area Unemployment Statistics
* Small Area Income and Poverty Estimates (SAIPE)

These datasets can be related using the FIPS code, which can indicate the county each row pertains to. All of these datasets I've imported contain a FIPS code. Therefore, I can join each of them using the county segment of the FIPS code.

I will explore their relations in the EDA section of this notebook. However, I will need to clean and merge the datasets first.

Please see more information regarding the datasets in the Jupyter notebook.

## Results

I was able to successfully create a Machine Learning model that can predict food insecurity rates for a given county in the United States and answer my project questions. I will demonstrate this with the following choropleth maps. To see how I reached these results, please run my notebook and walk through each step from data preparation/cleaning, EDA, to ML modeling.

The interactive graphs in the results section of my notebook answer the following project questions:

* *Which counties of the united states are most impacted by food insecurity, and what indicators like poverty, unemployment, and income are correlated to it?*
* *How can we predict food insecurity in the future?*

Please be sure to check out the interactive maps on the notebook, so you can view specific county food insecurity rates.

### 2020 Actual Food Insecurity Map

![alt text](https://github.com/IT4063C-Fall22/final-project-classaccounts/blob/main/images/2020_actual.png?raw=true)
### 2020 Machine Learning Model Predicted Food Insecurity Map

![alt text](https://github.com/IT4063C-Fall22/final-project-classaccounts/blob/main/images/2020_prediction.png?raw=true)
### 2020 Machine Learning Model Predicting Food Insecurity Map Without 2020 Training Data
Since the previous map was predicting 2020 actuals based on trained data from 2010 - 2020, I also wanted to create another trained model (2010 - 2019) that was not fitted with 2020 data and see how it performs predicting 2020 food insecurity, you can see the results below. However, it does tend to overestimate counties with low food insecurity and predicts them to be higher than their actual (dark blue). I believe this is due to the unemployment spike from the COVID pandemic (2020) where food insecurity followed the mean yearly trend while unemployment rose and in which training data relating to that was purposely removed from this model. The regression model above would do a better job of predicting food insecurity with the COVID unemployment spike.

![alt text](https://github.com/IT4063C-Fall22/final-project-classaccounts/blob/main/images/2020_prediction_no2020train.png?raw=true)