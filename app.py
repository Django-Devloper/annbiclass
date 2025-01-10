import streamlit as st 
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

scaller = StandardScaler()
model = load_model('ann_binary_classification.keras')

def leble_encoder_decoder(column_name,pickler_name,input_data):
    with open(pickler_name,'rb') as file:
        pickle_loader = pickle.load(file)
    input_data[column_name] = pickle_loader.transform([input_data[column_name]])[0]
    return input_data

def one_hot_encoder_loader(column_name,pickler_name,input_data):
    with open(pickler_name ,'rb') as file:
        pickle_loader = pickle.load(file)
    column_name_array  = pickle_loader.transform(input_data[[column_name]]).toarray()
    column_array_df = pd.DataFrame(column_name_array , columns=pickle_loader.get_feature_names_out([column_name]))
    input_data = pd.concat([input_data.drop(column_name,axis=1),column_array_df],axis=1)
    return input_data

lable_pickeler = {'Exercise':'Exerciseencoder.pkl','Skin_Cancer':'Skin_Cancerencoder.pkl','Arthritis':'Arthritisencoder.pkl','Depression':'Depressionencoder.pkl',
               'Other_Cancer':'Other_Cancerencoder.pkl','Sex':'Sexencoder.pkl','Smoking_History':'Smoking_Historyencoder.pkl'
               }

one_hot_pickler = {'General_Health' : 'General_Healthencoder.pkl', 'Checkup' : 'Checkupencoder.pkl', 'Diabetes' : 'Diabetesencoder.pkl',
    'Age_Category' : 'Age_Categoryencoder.pkl'}

General_Health_options = ['Poor', 'Very Good', 'Good', 'Fair', 'Excellent']
Checkup_options = ['Within the past 2 years', 'Within the past year','5 or more years ago', 'Within the past 5 years', 'Never']
Diabetes_options = ['No', 'Yes', 'No, pre-diabetes or borderline diabetes','Yes, but female told only during pregnancy']
Binary_options = ['Yes' ,'No']
Age_Category_options = ['70-74', '60-64', '75-79', '80+', '65-69', '50-54', '45-49', '18-24', '30-34', '55-59', '35-39', '40-44', '25-29']

st.title("Heart Attack Prediaction ")
General_Health_options = st.selectbox("Select General Health :", General_Health_options)
Checkup_options = st.selectbox("Select Checkup :", Checkup_options)
Exercise_options = st.radio('Do you Exercise ?' , Binary_options)
Skin_Cancer_options = st.radio("Do you suffer from Skin Cancer ? ", Binary_options)
Other_Cancer_options = st.radio("Do you suffer from Other Cancer ? ", Binary_options)
Depression_options = st.radio("Do you suffer from Depression ?:", Binary_options)
Diabetes_options = st.selectbox("Select Diabetes  stsus:", Diabetes_options)
Arthritis_option = st.radio("Do you suffer from Arthritis ?", Binary_options)
Sex_option = st.selectbox("Select Sex :", ['Male' , 'Female'])
Age_Category_options = st.selectbox("Select Age Category :", Age_Category_options)
Height = st.number_input('Enter Height in CM')
Weight = st.number_input('Enter Weight in KG ')
bmi = st.number_input('Enter BMI ')
Smoking_History = st.radio("Do you have Smoking History :", Binary_options)
Alcohol_Consumption = st.slider("Alcohol Consumption in Litter/Month :", 0, 30)
Fruit_Consumption = st.slider("Fruit Consumption in Kg/Month :", 0, 120)
Green_Vegetables_Consumption = st.slider("Green Vegetables Consumption in Kg/Month :", 0, 130)
FriedPotato_Consumption = st.slider("Fried Potato Consumption in Kg/Month :", 0, 130)

analysis = st.button('Helth Analysis',use_container_width=True ,type='primary')
if analysis:
    input_data = {
	'General_Health' : General_Health_options,
	'Checkup' :	Checkup_options,
	'Exercise':	Exercise_options,
	'Skin_Cancer':Skin_Cancer_options,
	'Other_Cancer':	Other_Cancer_options,
	'Depression':Depression_options,
	'Diabetes':	Diabetes_options,
	'Arthritis':Arthritis_option,
	'Sex':Sex_option,
	'Age_Category': Age_Category_options,
	'Height_(cm)':	Height,
	'Weight_(kg)':	Weight,
	'BMI':		bmi,
	'Smoking_History':Smoking_History,
	'Alcohol_Consumption':	Alcohol_Consumption,
	'Fruit_Consumption':	Fruit_Consumption,
	'Green_Vegetables_Consumption': Green_Vegetables_Consumption,
	'FriedPotato_Consumption':	FriedPotato_Consumption
	}
    for column_name ,pickler_name in lable_pickeler.items():
        input_data = leble_encoder_decoder(column_name,pickler_name,input_data)

    input_data_df = pd.DataFrame([input_data])
    for column_name ,pickler_name in one_hot_pickler.items():
        input_data_df = one_hot_encoder_loader(column_name,pickler_name,input_data_df)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    scalled_data = scaler.transform(input_data_df)
    predict= model.predict(scalled_data)
    percentage = predict[0][0] * 100
    if percentage < 50 :
        st.info(f"No Heart Disease Detected. Predicted chance is : {percentage}")
    else:
        st.info(f"Heart Disease Detected. Predicted chance is : {percentage}")
