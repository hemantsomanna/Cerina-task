#!/usr/bin/env python
# coding: utf-8

# In[1]:


#TASK 1

#importing necessary libraries
import pandas as pd
from datetime import datetime

# Loading the dataset
file_path = r"C:\Users\heman\OneDrive\Desktop\Homestays_Data.csv"
df = pd.read_csv(file_path)

# Dropping unnecessary columns
df = df.drop(['thumbnail_url', 'cancellation_policy', 'cleaning_fee', 'host_has_profile_pic', 'neighbourhood', 'id','bed_type', 'cancellation_policy'], axis=1)

# Number of rows before dropping
total_rows_before = len(df)

# Dropping rows with missing values in 'host_since' and 'last_review' columns
df_cleaned = df.dropna(subset=['host_since', 'last_review'])

# Number of rows after dropping
total_rows_after = len(df_cleaned)

# Calculating the number of rows dropped due to missing values in 'host_since' and 'last_review' columns
rows_dropped_host_since = total_rows_before - len(df.dropna(subset=['host_since']))
rows_dropped_last_review = total_rows_before - len(df.dropna(subset=['last_review']))

# Counting missing amenities
missing_amenities_count = df_cleaned['amenities'].isnull().sum()

# Printing the number of rows dropped due to missing values in 'host_since', 'last_review' and 'amenities' columns
print("Number of rows dropped due to missing values in 'host_since':", rows_dropped_host_since)
print("Number of rows dropped due to missing values in 'last_review':", rows_dropped_last_review)
print("Number of rows dropped due to missing values in 'amenities:", missing_amenities_count)

# Calculating percentages
percent_dropped_host_since = (rows_dropped_host_since / total_rows_before) * 100
percent_dropped_last_review = (rows_dropped_last_review / total_rows_before) * 100

# Printing percentages
print("Percentage of rows dropped due to missing values in 'host_since': {:.2f}%".format(percent_dropped_host_since))
print("Percentage of rows dropped due to missing values in 'last_review': {:.2f}%".format(percent_dropped_last_review))

# Printing total number of rows left
print("Total number of rows left after dropping missing values:", total_rows_after)

# Converting 'host_since' column to datetime
df_cleaned['host_since'] = pd.to_datetime(df_cleaned['host_since'])

# Calculating host tenure in years
current_date = datetime.now()
df_cleaned['Host_Tenure'] = (current_date - df_cleaned['host_since']).dt.days / 365.25  # divide by 365.25 to account for leap years
df_cleaned['Host_Tenure'] = df_cleaned['Host_Tenure'].round()

# Defining function to count amenities
def count_amenities(amenities_list):
    return len(amenities_list.split(','))

# Applying function to calculate amenities count
df_cleaned['Amenities_Count'] = df_cleaned['amenities'].apply(lambda x: count_amenities(x))

# Converting 'last_review' column to datetime
df_cleaned['last_review'] = pd.to_datetime(df_cleaned['last_review'])

# Calculating days since last review
df_cleaned['Days_Since_Last_Review'] = (datetime.today() - df_cleaned['last_review']).dt.days

# Saving modified dataframe to CSV
df_cleaned.to_csv('modified_data.csv', index=False)
desktop_path = r"C:\Users\heman\OneDrive\Desktop"

# Saving modified dataframe to CSV on desktop
df_cleaned.to_csv(f"{desktop_path}\\modified_data.csv", index=False)


# In[2]:


#TASK 2

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


file_path = r"C:\Users\heman\OneDrive\Desktop\modified_data.csv"
df = pd.read_csv(file_path)

# Selecting relevant columns
selected_columns = ['log_price', 'room_type', 'property_type', 'instant_bookable',
                    'accommodates', 'bathrooms', 'number_of_reviews', 'review_scores_rating',
                    'bedrooms', 'beds','Host_Tenure', 'Amenities_Count', 'Days_Since_Last_Review']
df = df[selected_columns]

# One-hot encode the three categorical columns
df = pd.get_dummies(df, columns=['room_type', 'property_type', 'instant_bookable'])

# Defining Correlation Matrix
correlation_matrix = df.corr()

# Visualize correlation matrix
plt.figure(figsize=(16, 16))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".1f")  # Updated fmt parameter
plt.title('Correlation Matrix')
plt.show()

# Scatter plots for numerical features vs log_price
plt.figure(figsize=(16, 10))
numeric_features = ['accommodates', 'bathrooms', 'number_of_reviews', 'review_scores_rating',
                    'bedrooms', 'beds', 'Host_Tenure', 'Amenities_Count', 'Days_Since_Last_Review']
for i, feature in enumerate(numeric_features, 1):
    plt.subplot(3, 3, i)
    sns.scatterplot(x=feature, y='log_price', data=df)
    plt.title(f'log_price vs {feature}')

plt.tight_layout()
plt.show()

file_path = r"C:\Users\heman\OneDrive\Desktop\modified_data.csv"
df = pd.read_csv(file_path)

# Box plots for categorical features vs log_price

#for room_type
plt.figure(figsize=(10, 6))
sns.boxplot(x='room_type', y='log_price', data=df)
plt.title('log_price by Room Type')
plt.xlabel('Room Type')
plt.ylabel('log_price')

#for property_type
plt.figure(figsize=(10, 6))
sns.boxplot(x='property_type', y='log_price', data=df)
plt.title('log_price by Property Type')
plt.xlabel('Property Type')
plt.xticks(rotation=45)
plt.ylabel('log_price')

#for instant_bookable
plt.figure(figsize=(10, 6))
sns.boxplot(x='instant_bookable', y='log_price', data=df)
plt.title('log_price by instant_bookable')
plt.xlabel('instant_bookable')
plt.ylabel('log_price')

plt.tight_layout()
plt.show()


# In[3]:


#TASK 3

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"C:\Users\heman\OneDrive\Desktop\Homestays_Data.csv"
df = pd.read_csv(file_path)

# Creating a GeoDataFrame from latitude and longitude columns
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))

# Loading the world map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Filtering  the world map to include only North America since the data set provided is only in North America
north_america = world[(world['continent'] == 'North America')]

# Creating a subplot with the North America map
fig, ax = plt.subplots(figsize=(10, 10))
north_america.plot(ax=ax, color='lightgrey')

# Scatter plot of the homestay locations in North America, color-coded by log_price
scatter = gdf.plot(ax=ax, column='log_price', cmap='coolwarm', legend=True, legend_kwds={'label': 'Log Price'})

#Naming the x-axis, y-axis and the visualisation
plt.title('Homestay Locations in North America (Color Graded by Log Price)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# In[4]:


#TASK 4

import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Loading the dataset
data = pd.read_csv("C:\\Users\\heman\\OneDrive\\Desktop\\Homestays_Data.csv")

#Using TextBlob to perform sentiment analysis on the description column which returns polarity i.e positive or negative
def get_sentiment_score(text):
    polarity = TextBlob(text).sentiment.polarity
    return polarity

# Applying sentiment analysis and adding a column for sentiment score
data['sentiment_score'] = data['description'].apply(get_sentiment_score)

# Counting positive and negative descriptions
positive_count = (data['sentiment_score'] > 0).sum()
negative_count = (data['sentiment_score'] < 0).sum()

#Printing
print("Number of Positive Descriptions:", positive_count)
print("Number of Negative Descriptions:", negative_count)

# Tokenizing descriptions and removing stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
positive_descriptions = ' '.join(data[data['sentiment_score'] > 0]['description'])
negative_descriptions = ' '.join(data[data['sentiment_score'] < 0]['description'])

positive_tokens = nltk.word_tokenize(positive_descriptions)
negative_tokens = nltk.word_tokenize(negative_descriptions)

positive_words = [word.lower() for word in positive_tokens if word.isalpha() and word.lower() not in stop_words]
negative_words = [word.lower() for word in negative_tokens if word.isalpha() and word.lower() not in stop_words]

# Getting the top 5 words repeated in positive and negative descriptions and printing them
top_positive_words = Counter(positive_words).most_common(5)
top_negative_words = Counter(negative_words).most_common(5)

print("\nTop 5 Words in Positive Descriptions:")
for word, count in top_positive_words:
    print(word, "-", count)

print("\nTop 5 Words in Negative Descriptions:")
for word, count in top_negative_words:
    print(word, "-", count)

#visualization of the relationship between sentiment score and log_price
plt.figure(figsize=(10, 6))
plt.hexbin(x=data['sentiment_score'], y=data['log_price'], gridsize=50, cmap='Blues', mincnt=1)
plt.colorbar(label='Density')
plt.xlabel('Sentiment Score')
plt.ylabel('log_price')
plt.title('Relationship between Sentiment Score and log_price')
plt.show()


# In[11]:


#TASK 5

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"C:\Users\heman\OneDrive\Desktop\Homestays_Data.csv"
df = pd.read_csv(file_path)

# Parsing the 'amenities' column
df['amenities_list'] = df['amenities'].apply(lambda x: x.strip('{}').replace('"', '').split(','))

# Creating binary features for each amenity to handle each of them seperately
amenity_columns = df['amenities_list'].explode().str.get_dummies().columns
df[amenity_columns] = pd.DataFrame(df['amenities_list'].apply(lambda x: [1 if amenity in x else 0 for amenity in amenity_columns]).tolist(), columns=amenity_columns, index=df.index)

# Calculating correlation between amenities and log_price
amenity_price_correlation = df[amenity_columns].apply(lambda x: x.corr(df['log_price']))

#Conduct two-sample t-tests for each amenity and calculate p-value
amenity_p_values = df[amenity_columns].apply(lambda x: stats.ttest_ind(df[df[x.name] == 1]['log_price'], df[df[x.name] == 0]['log_price']).pvalue)

# Combining the results into a DataFrame
amenity_analysis = pd.DataFrame({'Correlation': amenity_price_correlation, 'P-value': amenity_p_values})

# Sorting in descending order of correlation
amenity_analysis = amenity_analysis.sort_values(by='Correlation', ascending=False)

# Visualising the correlation of each amenity and log_price
plt.figure(figsize=(20, 8))
ax = amenity_analysis['Correlation'].plot(kind='bar')
plt.title('Correlation between Amenities and log_price')
plt.xlabel('Amenity')
plt.ylabel('Correlation')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.show()

# Printing top 5 amenities associated with higher log_price
print("Top 5 Amenities Associated with Higher log_price:")
print(amenity_analysis.head(5))

# Printing bottom 5 amenities associated with lower log_price
print("\nBottom 5 Amenities Associated with Lower log_price:")
print(amenity_analysis.tail(5))


# In[12]:


#TASK 6

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os

# Load the dataset
file_path = r"C:\Users\heman\OneDrive\Desktop\modified_data.csv"
df = pd.read_csv(file_path)

# Initializing OneHotEncoder
onehot_encoder = OneHotEncoder()

# Applying OneHotEncoder to 'room_type', 'city', and 'property_type' columns
room_type_encoded = onehot_encoder.fit_transform(df[['room_type']])
city_encoded = onehot_encoder.fit_transform(df[['city']])
property_type_encoded = onehot_encoder.fit_transform(df[['property_type']])

# Get the unique values present in 'room_type', 'city', and 'property_type' columns
room_type_values = df['room_type'].unique()
city_values = df['city'].unique()
property_type_values = df['property_type'].unique()

# Convert the encoded data to be able to add in the dataset
room_type_df = pd.DataFrame(room_type_encoded.toarray(), columns=room_type_values)
city_df = pd.DataFrame(city_encoded.toarray(), columns=city_values)
property_type_df = pd.DataFrame(property_type_encoded.toarray(), columns=property_type_values)

#making new columns in the dataset for model building
df = pd.concat([df, room_type_df, city_df, property_type_df], axis=1)

# Dropping the original columns
df.drop(['room_type', 'city', 'property_type'], axis=1, inplace=True)

# Saving the new dataset CSV file for model building 
final_file_name = "final_dataset.csv"
final_file_path = r"C:\Users\heman\OneDrive\Desktop\final_homestay_data.csv" 
df.to_csv(final_file_path, index=False)

print(df.head())


# In[7]:


#TASK 7

#As the dataset was already cleaned and preprocessed I could use the final_homestay_dataset.
#For model building I have used the scikit-learn library which has all the complex models that I have utilised.
#As a starting point model, linear regression was selected because of its ease of interpretation and simplicity.
#Two ensemble techniques with a reputation for managing complex interactions and offering excellent forecast accuracy are Random Forest and Gradient Boosting.
#Support Vector Machine: Capable of capturing intricate patterns, it works well in high-dimensional spaces.
#Neural Network (MLP): Able to discover intricate nonlinear patterns in data.
#K-Nearest Neighbors: This straightforward method works well for regression applications, particularly when the data clearly shows local trends.
#XGBoost: A potent gradient boosting algorithm with a solid reputation for accuracy and efficiency.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = r"C:\Users\heman\OneDrive\Desktop\final_homestay_data.csv"
df = pd.read_csv(file_path)

# Selecting only the numerical columns and dropping rows with missing values
numerical_columns = df.select_dtypes(include='number').columns
df = df[numerical_columns].dropna()

# Selecting features and target variable for the model
X = df.drop(['log_price'], axis=1)
y = df['log_price']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the linear regression model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

#Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Linear Regression:")
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Initializing Random Forest and Gradient Boosting regressors
rf_model = RandomForestRegressor(random_state=42)
gb_model = GradientBoostingRegressor(random_state=42)

# Training models
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Making predictions
rf_y_pred = rf_model.predict(X_test)
gb_y_pred = gb_model.predict(X_test)

# Evaluating and printing MSE and R square for both the models
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)
print("\nRandom Forest:")
print("Mean Squared Error:", rf_mse)
print("R^2 Score:", rf_r2)

gb_mse = mean_squared_error(y_test, gb_y_pred)
gb_r2 = r2_score(y_test, gb_y_pred)
print("\nGradient Boosting:")
print("Mean Squared Error:", gb_mse)
print("R^2 Score:", gb_r2)

# Support Vector Machine regressor
svm_model = SVR()

svm_model.fit(X_train, y_train)

svm_y_pred = svm_model.predict(X_test)

# Evaluating SVM model
svm_mse = mean_squared_error(y_test, svm_y_pred)
svm_r2 = r2_score(y_test, svm_y_pred)
print("\nSupport Vector Machines:")
print("Mean Squared Error:", svm_mse)
print("R^2 Score:", svm_r2)

# Multilayer Perceptron regressor
mlp_model = MLPRegressor(random_state=42)

mlp_model.fit(X_train, y_train)

mlp_y_pred = mlp_model.predict(X_test)

# Evaluating MLP model
mlp_mse = mean_squared_error(y_test, mlp_y_pred)
mlp_r2 = r2_score(y_test, mlp_y_pred)
print("\nNeural Networks (MLP):")
print("Mean Squared Error:", mlp_mse)
print("R^2 Score:", mlp_r2)

# K-Nearest Neighbors regressor
knn_model = KNeighborsRegressor()

knn_model.fit(X_train, y_train)

knn_y_pred = knn_model.predict(X_test)

# Evaluating KNN model
knn_mse = mean_squared_error(y_test, knn_y_pred)
knn_r2 = r2_score(y_test, knn_y_pred)
print("\nK-Nearest Neighbors (KNN):")
print("Mean Squared Error:", knn_mse)
print("R^2 Score:", knn_r2)

# XGBoost regressor
xgb_model = XGBRegressor(random_state=42)

xgb_model.fit(X_train, y_train)

xgb_y_pred = xgb_model.predict(X_test)

# Evaluating XGBoost model
xgb_mse = mean_squared_error(y_test, xgb_y_pred)
xgb_r2 = r2_score(y_test, xgb_y_pred)
print("\nXGBoost:")
print("Mean Squared Error:", xgb_mse)
print("R^2 Score:", xgb_r2)

# Creating a dictionary to store model names and their corresponding R^2 scores
model_scores = {
    "Linear Regression": r2,
    "Random Forest": rf_r2,
    "Gradient Boosting": gb_r2,
    "Support Vector Machines": svm_r2,
    "Neural Networks (MLP)": mlp_r2,
    "K-Nearest Neighbors (KNN)": knn_r2,
    "XGBoost": xgb_r2
}

# Sorting the models based on their R^2 scores in descending order
sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

# Printing the order of best to worst working models
print("\nOrder of Best Working Models (based on R^2 score):")
for model_name, r2_score in sorted_models:
    print(f"{model_name}: {r2_score}")

# Printing the features being used from the dataset    
print("\nColumns being used as features:")
print(X.columns)



# In[8]:


#TASK 8

# Since the XGBoost, Random Forest and Gradient Boosting models worked the best respectively I'll be working on these three models in this task.

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# Initializing the three models
rf_model = RandomForestRegressor(random_state=42)
gb_model = GradientBoostingRegressor(random_state=42)
xgb_model = XGBRegressor(random_state=42)
lr_model = LinearRegression()

# Defining parameters for each model
rf_param_grid = {
    'n_estimators': [100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

gb_param_grid = {
    'n_estimators': [100],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'learning_rate': [0.1, 0.01]
}

xgb_param_grid = {
    'n_estimators': [100],
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.01]
}

# Initializing GridSearchCV (I've used smaller parameters in grids and CV value to save execution time)
rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=3, scoring='r2')
rf_grid_search.fit(X_train, y_train)

gb_grid_search = GridSearchCV(estimator=gb_model, param_grid=gb_param_grid, cv=3, scoring='r2')
gb_grid_search.fit(X_train, y_train)

xgb_grid_search = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid, cv=3, scoring='r2')
xgb_grid_search.fit(X_train, y_train)

# Getting the best parameters
best_params_rf = rf_grid_search.best_params_
best_params_gb = gb_grid_search.best_params_
best_params_xgb = xgb_grid_search.best_params_

# Training the models with the best parameters
rf_model_best = RandomForestRegressor(**best_params_rf, random_state=42)
rf_model_best.fit(X_train, y_train)

gb_model_best = GradientBoostingRegressor(**best_params_gb, random_state=42)
gb_model_best.fit(X_train, y_train)

xgb_model_best = XGBRegressor(**best_params_xgb, random_state=42)
xgb_model_best.fit(X_train, y_train)

# Evaluating the models
rf_r2_best = rf_model_best.score(X_test, y_test)
gb_r2_best = gb_model_best.score(X_test, y_test)
xgb_r2_best = xgb_model_best.score(X_test, y_test)

# Printing the best R square scores
print("Best R^2 Score (Random Forest):", rf_r2_best)
print("Best R^2 Score (Gradient Boosting):", gb_r2_best)
print("Best R^2 Score (XGBoost):", xgb_r2_best)


# In[9]:


#TASK 9

# My laptop's specs prevented me from applying SHAP values because it couldn't run it quickly and required high computation power. 
# I'm using the feature importance score by itself to visualize.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import shap
import numpy as np
import matplotlib.pyplot as plt

# Analyzing feature importances for tree-based models( Random Forest and Gradient Boosting)
print("Feature Importances:")
for model_name, model in [("Random Forest", rf_model_best), ("Gradient Boosting", gb_model_best)]:
    feature_importances = model.feature_importances_
    print(f"\n{model_name}:")
    for feature, importance in zip(X.columns, feature_importances):
        print(f"{feature}: {importance:.4f}")
    
    top_10_indices = np.argsort(feature_importances)[::-1][:10]
    
    top_10_features = [X.columns[i] for i in top_10_indices]
    top_10_importances = [feature_importances[i] for i in top_10_indices]

    # Plotting feature importances for top 10 features
    plt.figure(figsize=(10, 6))
    plt.barh(top_10_features, top_10_importances)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(f'Top 10 Feature Importances ({model_name})')
    plt.show()

# Analyzing feature importances for XGBoost
print("\nFeature Importances (XGBoost):")
feature_importances_xgb = xgb_model_best.feature_importances_
for feature, importance in zip(X.columns, feature_importances_xgb):
    print(f"{feature}: {importance:.4f}")

top_10_indices_xgb = np.argsort(feature_importances_xgb)[::-1][:10]

top_10_features_xgb = [X.columns[i] for i in top_10_indices_xgb]
top_10_importances_xgb = [feature_importances_xgb[i] for i in top_10_indices_xgb]

# Plotting feature importances for top 10 features for XGBoost
plt.figure(figsize=(10, 6))
plt.barh(top_10_features_xgb, top_10_importances_xgb)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top 10 Feature Importances (XGBoost)')
plt.show()


# In[10]:


#TASK 10

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Calculating the residuals
residuals = y_test - y_pred

# Calculating RMSE and R-squared values
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Printing RMSE and R-squared
print("Root Mean Squared Error (RMSE):", rmse)
print("\nR-squared (R2):", r2)

# Visualizing Residuals vs Predicted values
# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5, color='b')  # Scatter plot of predicted values vs. residuals
plt.axhline(y=0, color='r', linestyle='--')  # Add horizontal line at y=0
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

# Histogram
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.grid(True)
plt.show()

# Checking mean of the residuals calculated
mean_residuals = np.mean(residuals)
print("\nMean of Residuals:", mean_residuals)

# Checking for symmetry in residuals distribution
symmetry_check = "\nResiduals distribution appears to be symmetric around zero."
if mean_residuals < 0.05:
    symmetry_check += " \n(Mean is close to zero)"
print(symmetry_check)

# Checking for outliers
outliers = np.sum(np.abs(residuals) > 3 * np.std(residuals))
print("\nNumber of Outliers:", outliers)

# Checking for patterns in residuals
pattern_check = "\nNo discernible pattern in residuals across data points."
if np.max(np.abs(residuals)) > 1.96 * np.std(residuals):
    pattern_check = "\nThere may be a pattern in residuals across data points."
print(pattern_check)

# Checking for homoscedasticity
homoscedasticity_check = "\nResiduals appear to be homoscedastic (consistent spread)."
if np.var(residuals) > 1.5 * np.mean(residuals):
    homoscedasticity_check = "\nResiduals may exhibit heteroscedasticity (varying spread)."
print(homoscedasticity_check)

# Checking for normality of residuals
from scipy.stats import shapiro

# Doing the Shapiro-Wilk test for normality
statistic, p_value = shapiro(residuals)
normality_check = "\nResiduals appear to be normally distributed." if p_value > 0.05 else "\nResiduals do not appear to be normally distributed."
print(normality_check)

