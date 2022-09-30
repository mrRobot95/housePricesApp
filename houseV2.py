import streamlit as st
import numpy as np
import pandas as pd 
import pickle

## || LOAD PKL FILES FOR THE ENCODING PART
OldNew_encoder=pickle.load(open("OldNew_encoder.pkl",'rb'))
scaler=pickle.load(open("scaler.pkl",'rb'))
## || LOAD THE MODEL PKL FILE 
model=pickle.load(open("best_model.pkl",'rb'))

#------------------------------------() APPLICATION MAIN INTERFACE/////// 
st.write("""
# MSDE4 : ESTIMATION DE PRIX DE MAISONS

""")
st.sidebar.header('Please Insert the fields below :')

OldNew=st.sidebar.selectbox(" the oldeness of the house N:New Y:OLD " , ['N','Y'])
Year=st.sidebar.selectbox(" Year " ,[2021])
Month = st.sidebar.selectbox(" Month " ,[1,2,3,4,5,6,7,8,9,10,11,12])
#st.number_input("Month",min_value=1,max_value=12,format='%d',step=1.0) 
Property_Type = st.sidebar.selectbox('Property_Type', ['D', 'F', 'S','T'])

def house_input_features():
    data = {'Old/New': OldNew,
            'Year': Year,
            'Month': Month,
            'Property_Type': Property_Type }
    house = pd.DataFrame(data, index=[0])
    Property_data = {'D':[0],'F':[0],'S':[0],'T':[0]}
    Property_data[Property_Type][0]=1
    Property_data=pd.DataFrame(Property_data)
    house=pd.concat([house.drop(columns='Property_Type'),Property_data],axis="columns")
    return house

def Encoding_inputs(house):
    house['Old/New']=OldNew_encoder.transform(house['Old/New']) 
    house=scaler.transform(house)
    return house

def Estimate_the_Price_Button():
    st.subheader('selected house attributes :')
    house_features = house_input_features()
    house = house_features
    st.write(house)
    house=Encoding_inputs(house)
    prediction = np.trunc(np.exp(model.predict(house)))
    st.subheader("Estimated Price")
    st.write(prediction)

# THE CLICK ON THE BUTTON Script    
if st.sidebar.button('Estimate the House Price'):
    Estimate_the_Price_Button()



