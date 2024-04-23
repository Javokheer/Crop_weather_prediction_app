import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="ML Predictions", page_icon="🧠", layout="wide")

# Function to load data
def load_data():
    # Modify the path according to your file location
    df = pd.read_excel('Projected_impacts_datasheet_11.24.2021.xlsx')
    df.columns = df.columns.str.replace(' ', '_')
    # Dictionary with new column names
    new_column_names = {
        'Ref_No_': 'Ref_No',
        'Current_Annual_Precipitation_(mm)__area_weighted_': 'Current_Annual_Precipitation_(mm)__area_weighted',
        'Global_delta_T_from_pre-industrial_period_': 'Global_delta_T_from_pre-industrial_period',
        '_Annual_Precipitation_change__from_2005_(mm)': 'Annual_Precipitation_change__from_2005_(mm)',
        'Others_': 'Others',
        '_Methods':'Method'
    }

    # Rename columns
    df.rename(columns=new_column_names, inplace=True)
    df.drop(columns=['Reference', 'Publication_year', 'Ref_No', 'Method', 'Scenario_source', 'doi', 'ID'], inplace=True)
    df = df[['Local_delta_T_from_2005', 'Global_delta_T_from_2005', 'latitude', 'longitude', 'Climate_scenario', 'Future_Mid-point', 'Current_Average_Temperature_(dC)_area_weighted', 'Country', 'Time_slice', 'Fertiliser', 'Irrigation', 'Cultivar', 'Adaptation_type']]

    return df

# Function to encode categorical columns
def encode_columns(df, columns):
    label_encoder = LabelEncoder()
    for column in columns:
        df[column] = label_encoder.fit_transform(df[column])
    return df

# Streamlit app starts here
st.title("Climate Impact Prediction Web App")

df = load_data()
# Encoding categorical and boolean columns
# Adjusted to filter for object, category, and bool types as examples
categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns
categorical_cols = st.multiselect('Select categorical variables to encode', categorical_columns, default=[])

df = encode_columns(df, categorical_cols)

# User inputs for selecting dependent and independent variables
dependent_variable = st.selectbox('Select the dependent variable', df.columns)
numeric_columns = df.select_dtypes(include=[np.number]).columns
independent_variables = st.multiselect('Select independent variables', numeric_columns, default=numeric_columns.difference([dependent_variable]).tolist())


# Filtering the DataFrame based on user selection
DATA = df[independent_variables + [dependent_variable]]


# Dropping rows with NaN values
DATA = DATA.dropna()

# Train-test split
X = DATA.drop(dependent_variable, axis=1)
y = DATA[dependent_variable]

# Random Forest Model
st.write("Training the Random Forest Model...")
model = RandomForestRegressor(n_estimators=500, random_state=42)
model.fit(X, y)

# Displaying MSE and Feature Importance
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
st.write(f"Mean Squared Error: {mse}")

feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
st.write("Feature Importance:", feature_importance.to_frame())

# # Prediction Section
# st.write("## Make Predictions")
# input_data = {feature: st.number_input(f"Input {feature}", value=float(X[feature].mean())) for feature in X.columns}
# input_df = pd.DataFrame([input_data])

# # Button to make predictions
# if st.button('Predict'):
#     prediction = model.predict(input_df)
#     st.write(f"Predicted {dependent_variable}: {prediction[0]}")

# Prediction Section
st.write("## Make Predictions")

# Initialize an empty dictionary to store user inputs
input_data = {}

# Use a form to collect all inputs before submission
with st.form("prediction_form"):
    # Create a number input for each feature, storing them in the dictionary
    for feature in X.columns:
        # Store each input by feature name in the dictionary
        input_data[feature] = st.number_input(f"Input {feature}", value=float(X[feature].mean()))
    
    # Create a 'Predict' button inside the form
    submit_button = st.form_submit_button("Predict")

if submit_button:
    # Convert the input data to DataFrame only after the user clicks 'Predict'
    input_df = pd.DataFrame([input_data])
    
    # Perform the prediction
    prediction = model.predict(input_df)
    
    # Display the prediction result
    st.write(f"Predicted {dependent_variable}: {prediction[0]}")
