import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    plot_confusion_matrix,
    precision_recall_curve,
    roc_curve,
    mean_squared_error
)
import joblib

data = pd.read_csv("data/parking_violation_geocoded.csv")
data.head()
data.columns
data = data.drop(["summons_image"],axis=1)
data.info()
data['issue_date'] = pd.to_datetime(data['issue_date'], errors='coerce')
data['issue_date'] = data['issue_date'].dt.strftime('%Y-%m-%d')
data_copy = data.copy()
for i in data.columns :
    print(i, " : ",data[i].isna().sum())
    print("*" * 50)
data = data.drop(["violation_description","vehicle_year","violation_location","issuer_command","issuer_squad","unregistered_vehicle","meter_number","violation_legal_code","violation_status","violation_post_code"],axis=1)
data.shape
data.eq(0).sum()
for i in data.columns:
    if(data[i].isna().sum() > 0 ) : 
        print(i, ": ",data[i].isna().sum())
        print("*" * 50)
data = data.dropna()
data.shape
data.info()
violation_data = pd.read_excel("data/ParkingViolationCodes_January2020.xlsx")
violation_data.head()
violation_data.columns
data = data.merge(violation_data, left_on='violation_code',  right_on="VIOLATION CODE", how="left")
data["Actual_Fine_Amount"] = data.apply(
   lambda row: row.loc['Manhattan  96th St. & below\n(Fine Amount $)'] if row['Violation County'] == 'Manhattan' else row.loc['All Other Areas\n(Fine Amount $)'],
    axis=1
)
data.drop(['VIOLATION CODE','VIOLATION DESCRIPTION','Manhattan  96th St. & below\n(Fine Amount $)','All Other Areas\n(Fine Amount $)'], axis=1, inplace=True)
data['Diff_Fine_Amount'] = data['Actual_Fine_Amount'] - data['fine_amount']
data = data.drop(['violation','address','street_name','intersecting_street'],axis=1)
data.info()
cat_columns = ['state','license_type','issuing_agency','county','violation_county','law_section','sub_division','Violation County','vehicle_make','vehicle_body_type']
for i in cat_columns:
    print(i," : ",data[i].value_counts())
    print("*" * 50)
data = data.drop(['county','violation_county','plate','judgment_entry_date','vehicle_make','vehicle_body_type'],axis = 1)
data.rename(columns={'Violation County' : 'violation_county','VIOLATION CODE': 'Violation Code'}, inplace=True)
data.eq(0).sum()
data = data.drop(['precinct','violation_precinct','issuer_precinct','issuer_code','summons_number'],axis=1)
time_bins = [0, 360, 720, 1080, 1440]
time_labels = ['Night', 'Morning', 'Afternoon', 'Evening']
data['violation_time_formatted'] = pd.to_datetime(data['violation_time'], format='%H:%M').dt.hour * 60 + pd.to_datetime(data['violation_time'], format='%H:%M').dt.minute
data['TimeCategory'] = pd.cut(data['violation_time_formatted'], bins=time_bins, labels=time_labels, right=False)
data = data.drop(['violation_time_formatted'],axis=1)

data['issue_date_time'] = pd.to_datetime(data['issue_date'])

if pd.api.types.is_datetime64_any_dtype(data['issue_date_time']):
    data['DayOfWeek'] = data['issue_date_time'].dt.day_name()
else:
    print("The 'issue_date' column is not in a datetime format.")
data.describe().T
data.info()
data['reduced_fine_amount'] = data['reduction_amount'] / data['TotalAmount']
data.drop(data[data['Diff_Fine_Amount'] > 0].index, inplace=True)
data = data.drop(['state','Actual_Fine_Amount','Diff_Fine_Amount','issue_date_time','sub_division','issue_date','violation_time'],axis=1)
cat_columns = data.describe(include=["object"]).columns 
for i in cat_columns:
    print(i," : ",data[i].value_counts())
    print("*" * 50)
allowed_license_types = ["PAS", "OMT", "COM"]
data = data[data['license_type'].isin(allowed_license_types)]

allowed_issuing_agency = ["V", "T"]

data = data[data['issuing_agency'].isin(allowed_issuing_agency)]
data.info()

for i in cat_columns:
    print(i," : ",data[i].value_counts())
    print("*" * 50)
data.shape
num_columns = data.describe(include=["int","float"]).columns
cat_columns = data.describe(include=["object"]).columns 
data_encoded = pd.get_dummies(data, columns=cat_columns)
data_encoded.head()

data_encoded = data_encoded.drop(['TotalAmount','TimeCategory','reduced_fine_amount','lat','lon'], axis=1)  # Features
# data_encoded.to_csv('EDAData.csv', index=False)
X = data_encoded.drop(['reduction_amount'], axis=1)  # Features
y = data_encoded['reduction_amount']  # Target variable

print(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [10, 50, 100, 150, 200]
}

rf_model = RandomForestRegressor()
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_n_estimators = grid_search.best_params_['n_estimators']
best_score = -grid_search.best_score_

best_rf = RandomForestRegressor(n_estimators=best_n_estimators)
best_rf.fit(X_train, y_train)

test_predictions = best_rf.predict(X_test)
mse = mean_squared_error(y_test, test_predictions)
print(f"Best n_estimators: {best_n_estimators}")
print(f"Test Mean Squared Error: {mse}")
# data.to_csv('EDADataWithoutEncoding.csv', index=False)
joblib.dump(best_rf, "besy.sav.gz")