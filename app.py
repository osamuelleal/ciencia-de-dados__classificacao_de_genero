#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np


def predict_quality(model, df):
    
    predictions_data = predict_model(estimator = model, data = df)
    
    return predictions_data['Label'][0]
    
model = load_model('qda_model')


st.title('Classificando se o indivíduo é Masculino ou Feminino')
st.write('Esse aplicativo irá predizer se as características que você inserir pertencem à um homem ou uma mulher.         Preencha os campos abaixo e clique em "Predizer".')


altura = st.slider(label = 'Altura', min_value = 0.0,
                          max_value = 5.0 ,
                          value = 1.7,
                          step = 0.01)

peso = st.slider(label = 'Peso', min_value = 0.00,
                          max_value = 300.00 ,
                          value = 80.00,
                          step = 0.01)
                          

features = {'altura': altura, 'peso': peso}
 

features_df  = pd.DataFrame([features])

st.table(features_df)  

if st.button('Predizer'):
    
    prediction = predict_quality(model, features_df)
    
    st.write('Baseado nas características que você passou, o indivíduo é '+ str(prediction))

