
import streamlit as st
import pandas as pd
import joblib

model = joblib.load('fake_account_classifier.pkl')
label_encoders = joblib.load('label_encoders.pkl')

def predict_fake_account(input_data):
    input_df = pd.DataFrame(input_data)

    categorical_cols = ['profile_pic', 'sim_name_username', 'extern_url', 'private']
    for col in categorical_cols:
        input_df[col] = label_encoders[col].transform(input_df[col])

    feature_columns = ['profile_pic', 'ratio_numlen_username', 'len_fullname', 
                       'ratio_numlen_fullname', 'sim_name_username', 
                       'len_desc', 'extern_url', 'private', 
                       'num_posts', 'num_followers', 'num_following']
    
    input_df = input_df[feature_columns]

    prediction = model.predict(input_df)
    return "The account is likely fake." if prediction[0] == 1 else "The account is likely real."

st.title("Fake Account Detection")

profile_pic = st.selectbox("Profile Picture", ["Yes", "No"])
ratio_numlen_username = st.number_input("Ratio of Numeric Characters in Username", min_value=0.0, max_value=1.0)
len_fullname = st.number_input("Length of Full Name", min_value=0)
ratio_numlen_fullname = st.number_input("Ratio of Numeric Characters in Full Name", min_value=0.0, max_value=1.0)
sim_name_username = st.selectbox("Similarity of Name and Username", ["Full match", "Partial match", "No match"])
len_desc = st.number_input("Length of Description", min_value=0)
extern_url = st.selectbox("External URL", ["Yes", "No"])
private = st.selectbox("Account Privacy", ["Yes", "No"])
num_posts = st.number_input("Number of Posts", min_value=0)
num_followers = st.number_input("Number of Followers", min_value=0)
num_following = st.number_input("Number of Following", min_value=0)

user_input = {
    'profile_pic': [profile_pic],
    'ratio_numlen_username': [ratio_numlen_username],
    'len_fullname': [len_fullname],
    'ratio_numlen_fullname': [ratio_numlen_fullname],
    'sim_name_username': [sim_name_username],
    'len_desc': [len_desc],
    'extern_url': [extern_url],
    'private': [private],
    'num_posts': [num_posts],
    'num_followers': [num_followers],
    'num_following': [num_following]
}

if st.button("Predict"):
    result = predict_fake_account(user_input)
    st.write(result)
