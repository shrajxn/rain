import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Function to load and preprocess the data 
@st.cache_data
def load_data():
    data = pd.read_csv('C:\\Users\\Shrajan\\OneDrive\\Desktop\\rain\\weatherAUS.csv')
    features = [
        'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
        'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
        'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
        'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday'
    ]
    target = 'RainTomorrow'
    data = data[features + [target]].dropna()
    data[target] = data[target].apply(lambda x: 1 if x == 'Yes' else 0)
    X = data[features]
    y = data[target]
    return X, y

# Function to train the model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    numeric_features = [
        'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
        'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
        'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
        'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm'
    ]
    categorical_features = ['RainToday']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='No')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    return best_model, X_test, y_test

# Function to make predictions
def predict(model, input_data):
    prediction = model.predict(input_data)
    return 'Yes' if prediction[0] == 1 else 'No'

# Streamlit UI
st.title('Rainfall Prediction')

st.write('Enter the weather parameters to predict if it will rain tomorrow.')

MinTemp = st.number_input('MinTemp')
MaxTemp = st.number_input('MaxTemp')
Rainfall = st.number_input('Rainfall')
Evaporation = st.number_input('Evaporation')
Sunshine = st.number_input('Sunshine')
WindGustSpeed = st.number_input('WindGustSpeed')
WindSpeed9am = st.number_input('WindSpeed9am')
WindSpeed3pm = st.number_input('WindSpeed3pm')
Humidity9am = st.number_input('Humidity9am')
Humidity3pm = st.number_input('Humidity3pm')
Pressure9am = st.number_input('Pressure9am')
Pressure3pm = st.number_input('Pressure3pm')
Cloud9am = st.number_input('Cloud9am')
Cloud3pm = st.number_input('Cloud3pm')
Temp9am = st.number_input('Temp9am')
Temp3pm = st.number_input('Temp3pm')
RainToday = st.selectbox('RainToday', ['No', 'Yes'])

# Load data and train model
X, y = load_data()
model, X_test, y_test = train_model(X, y)

# Prepare input data for prediction
input_data = pd.DataFrame({
    'MinTemp': [MinTemp],
    'MaxTemp': [MaxTemp],
    'Rainfall': [Rainfall],
    'Evaporation': [Evaporation],
    'Sunshine': [Sunshine],
    'WindGustSpeed': [WindGustSpeed],
    'WindSpeed9am': [WindSpeed9am],
    'WindSpeed3pm': [WindSpeed3pm],
    'Humidity9am': [Humidity9am],
    'Humidity3pm': [Humidity3pm],
    'Pressure9am': [Pressure9am],
    'Pressure3pm': [Pressure3pm],
    'Cloud9am': [Cloud9am],
    'Cloud3pm': [Cloud3pm],
    'Temp9am': [Temp9am],
    'Temp3pm': [Temp3pm],
    'RainToday': [RainToday]
})

# Predict and display result
if st.button('Predict'):
    result = predict(model, input_data)
    st.write(f'Will it rain tomorrow? {result}')
