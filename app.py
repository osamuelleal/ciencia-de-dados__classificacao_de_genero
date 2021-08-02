{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "44548deb-cbee-4feb-ac89-27c6fecd3ad1",
   "metadata": {},
   "outputs": [],
   "source": [
    "from pycaret.classification import load_model, predict_model\n",
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "\n",
    "def predict_quality(model, df):\n",
    "    \n",
    "    predictions_data = predict_model(estimator = model, data = df)\n",
    "    \n",
    "    return predictions_data['Label'][0]\n",
    "    \n",
    "model = load_model('qda_model')\n",
    "\n",
    "\n",
    "st.title('Classificando se o indivíduo é Masculino ou Feminino')\n",
    "st.write('Esse aplicativo irá predizer se as características que você inserir pertencem à um homem ou uma mulher.\\\n",
    "         Preencha os campos abaixo e clique em \"Predizer\".')\n",
    "\n",
    "\n",
    "altura = st.sidebar.slider(label = 'Altura', min_value = 0.0,\n",
    "                          max_value = 5.0 ,\n",
    "                          value = 1.7,\n",
    "                          step = 0.1)\n",
    "\n",
    "peso = st.sidebar.slider(label = 'Peso', min_value = 0.00,\n",
    "                          max_value = 300.00 ,\n",
    "                          value = 80.00,\n",
    "                          step = 0.01)\n",
    "                          \n",
    "\n",
    "features = {'altura': altura, 'peso': peso}\n",
    " \n",
    "\n",
    "features_df  = pd.DataFrame([features])\n",
    "\n",
    "st.table(features_df)  \n",
    "\n",
    "if st.button('Predizer'):\n",
    "    \n",
    "    prediction = predict_quality(model, features_df)\n",
    "    \n",
    "    st.write('Baseado nas características que você passou, o indivíduo é '+ str(prediction))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
