import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

st.markdown("<style>.stApp{background-color:#0E1117; color:white;}</style>",unsafe_allow_html=True)


st.markdown("<h1 style='color: gold; text-align:center;'>EMAIL SPAM DETECTOR</h1>",unsafe_allow_html=True)
st.markdown("<br></br>",unsafe_allow_html=True)
df=pd.read_csv('mail_data.csv')

proc_data=df.fillna('')
proc_data['Category']=proc_data['Category'].map({'spam':0,'ham':1})
X=proc_data['Message']
Y=proc_data['Category']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3)

feature_extraction=TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)


X_train_features = feature_extraction.fit_transform(X_train)
X_test_features= feature_extraction.transform(X_test)


Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')

model=LogisticRegression()
model.fit(X_train_features,Y_train)
predict=model.predict(X_test_features)

st.markdown("<h4 style='color:white; font-weight: bold;'>ENTER EMAIL</h4>",unsafe_allow_html=True)
input_data = st.text_area(label='',
    placeholder="Paste the email message here...",
    height=200
)


st.markdown("""
<style>
div.stButton > button {
    background-color:skyblue;
    color: black;
    border-radius: 10px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

if (st.button(" Check Email ")):
  

    input_data_features=feature_extraction.transform([input_data])
    predict=model.predict(input_data_features)


    if (predict[0]==0):
        st.error("SPAM MAIL")
    else:
        st.success("HAM MAIL")
