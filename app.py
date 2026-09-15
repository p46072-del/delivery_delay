%%writefile streamlit_app.py

import streamlit as st
import joblib
import pandas as pd

# Load the trained model
logi = joblib.load('logi.sav')

st.title('Delivery Delay Prediction')

st.write("Enter the features below to predict delivery delay:")

# Input fields for each feature
delivery_distance = st.slider('Delivery Distance', min_value=0.0, max_value=200.0, value=20.0, step=0.1)
traffic_congestion = st.slider('Traffic Congestion (1-5)', min_value=1, max_value=5, value=3, step=1)
weather_condition = st.slider('Weather Condition (1-5)', min_value=1, max_value=5, value=3, step=1)
delivery_slot = st.slider('Delivery Slot (1-3)', min_value=1, max_value=3, value=2, step=1)
driver_experience = st.slider('Driver Experience (years)', min_value=0, max_value=30, value=5, step=1)
num_stops = st.slider('Number of Stops', min_value=0, max_value=15, value=2, step=1)
vehicle_age = st.slider('Vehicle Age (years)', min_value=0, max_value=20, value=3, step=1)
road_condition_score = st.slider('Road Condition Score (1-5)', min_value=1, max_value=5, value=3, step=1)
package_weight = st.slider('Package Weight (kg)', min_value=0.0, max_value=100.0, value=12.0, step=0.1)
fuel_efficiency = st.slider('Fuel Efficiency (km/L)', min_value=0.0, max_value=30.0, value=12.0, step=0.1)
warehouse_processing_time = st.slider('Warehouse Processing Time (minutes)', min_value=0, max_value=200, value=120, step=1)

# Create a DataFrame from the input values
input_data = pd.DataFrame([{
    'Delivery_Distance': delivery_distance,
    'Traffic_Congestion': traffic_congestion,
    'Weather_Condition': weather_condition,
    'Delivery_Slot': delivery_slot,
    'Driver_Experience': driver_experience,
    'Num_Stops': num_stops,
    'Vehicle_Age': vehicle_age,
    'Road_Condition_Score': road_condition_score,
    'Package_Weight': package_weight,
    'Fuel_Efficiency': fuel_efficiency,
    'Warehouse_Processing_Time': warehouse_processing_time
}])

if st.button('Predict Delivery Delay'):
    prediction = logi.predict(input_data)
    prediction_proba = logi.predict_proba(input_data)[0]

    st.subheader('Prediction:')
    if prediction[0] == 1:
        st.error('The model predicts a **Delivery Delay**.')
    else:
        st.success('The model predicts **No Delivery Delay**.')
    
    st.write(f"Probability of No Delay: {prediction_proba[0]:.2f}")
    st.write(f"Probability of Delay: {prediction_proba[1]:.2f}")
