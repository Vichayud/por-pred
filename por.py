#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import numpy as np

st.write("""
# Portuguese Grade Prediction

Predict the grade of protoguese student!
"""
)

st.sidebar.header('User Input Features')

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        #input
        age = st.sidebar.slider('Age', 15,22,16)
        absences = st.sidebar.slider('Absence (Number of Times)', 0,8,2)
        Medu = st.sidebar.selectbox("Mother's Education" , ('None','Primary','5-9th Grade','Secondary','Higher'))
        Fedu = st.sidebar.selectbox("Father's Education" , ('None','Primary','5-9th Grade','Secondary','Higher'))
        traveltime = st.sidebar.slider('Travel Time from Home to School (Hours)', 1,4,1)
        studytime = st.sidebar.slider('Weekly Study Time (Hours)', 1,4,2)
        failures = st.sidebar.slider('Number of Past Class Failure', 0,3,0)
        famrel = st.sidebar.slider('Quality of Family (1 = lowest)', 1,5,5)
        freetime = st.sidebar.slider('Free Time after School (1 = lowest)', 1,5,4)
        goout = st.sidebar.slider('Going out with friend (1 = lowest)', 1,5,4)
        Dalc = st.sidebar.slider('Weekday Alcohol Consumption (1 = lowest)', 1,5,1)
        Walc = st.sidebar.slider('Weekend Alcohol Consumption (1 = lowest)', 1,5,2)
        health = st.sidebar.slider('Health (1 = lowest)', 1,5,5)
        school = st.sidebar.selectbox("School" , ('GP','MS'))
        sex = st.sidebar.selectbox("Sex" , ('F','M'))
        address = st.sidebar.selectbox("Address" , ('Rural','Urban'))
        famsize = st.sidebar.selectbox("family size" , ('Higher than 3','3 or lower'))
        Pstatus = st.sidebar.selectbox("Parent's status" , ('Apart','Together'))
        Mjob = st.sidebar.selectbox("Mother's job" , ('At Home', 'Health', 'Other', 'Services', 'Teacher'))
        Fjob = st.sidebar.selectbox("Father's job" , ('At Home', 'Health', 'Other', 'Services', 'Teacher'))
        reason = st.sidebar.selectbox("Reason for choosing school" , ('Course', 'Close to Home', 'Other', 'Reputation'))
        guardian = st.sidebar.selectbox("Guardian" , ('Father','Mother','Other'))
        schoolsup = st.sidebar.selectbox("Extra Education Support" , ('No','Yes'))
        famsup = st.sidebar.selectbox("Family Education Support" , ('No','Yes'))
        paid = st.sidebar.selectbox("Extra Paid Class" , ('No','Yes'))
        activities = st.sidebar.selectbox("Extra-curricular Activities" , ('No','Yes'))
        nursery = st.sidebar.selectbox("Attended Nursery School" , ('No','Yes'))
        higher = st.sidebar.selectbox("Wants to Take Higher Education" , ('No','Yes'))
        internet = st.sidebar.selectbox("Internet Access at Home" , ('No','Yes'))
        romantic = st.sidebar.selectbox("In a Romantic Relationship" , ('No','Yes'))

        age = (age-16.65546218487395)/1.2664844134763047
        absences = (absences-6.3165266106442575)/8.176147670919773
    
        Medu_list = ['None','Primary','5-9th Grade','Secondary','Higher']
        Fede_list = ['None','Primary','5-9th Grade','Secondary','Higher']

        school_list = ['GP','MS']
        sex_list = ['F','M']
        address_list = ['Rural','Urban']
        famsize_list = ['Higher than 3','3 or lower']
        Pstatus_list = ['Apart','Together']
        Mjob_list = ['At Home', 'Health', 'Other', 'Services', 'Teacher']
        Fjob_list = ['At Home', 'Health', 'Other', 'Services', 'Teacher']
        reason_list = ['Course', 'Close to Home', 'Other', 'Reputation']
        guardian_list = ['Father','Mother','Other']
        no_yes = ['No','Yes']

        Medu = Medu_list.index(Medu)
        Fedu = Medu_list.index(Fedu)

        school_GP = np.where(school == 'GP',1,0)
        school_MS = np.where(school == 'GP',0,1)
        sex_F = np.where(sex == 'F',1,0)
        sex_M = np.where(sex == 'F',0,1)
        address_R = np.where(address == 'Rural',1,0)
        address_U = np.where(address == 'Urban',1,0)
        famsize_GT3 = np.where(famsize == 'Higher than 3',1,0)
        famsize_LE3 = np.where(famsize == 'Higher than 3',0,1)
        Pstatus_A = np.where(Pstatus == 'Apart',1,0) 
        Pstatus_T = np.where(Pstatus == 'Apart',0,1)
        Mjob_at_home = np.where(Mjob == 'At Home',1,0)
        Mjob_health = np.where(Mjob == 'Health',1,0)
        Mjob_other =  np.where(Mjob == 'Other',1,0)
        Mjob_services = np.where(Mjob == 'Services',1,0)
        Mjob_teacher = np.where(Mjob == 'Teacher',1,0)
        Fjob_at_home = np.where(Fjob == 'At Home',1,0)
        Fjob_health = np.where(Fjob == 'Health',1,0)
        Fjob_other =  np.where(Fjob == 'Other',1,0)
        Fjob_services = np.where(Fjob == 'Services',1,0)
        Fjob_teacher = np.where(Fjob == 'Teacher',1,0)
        reason_course = np.where(reason == 'Course',1,0)
        reason_home = np.where(reason == 'Close to Home',1,0)
        reason_other = np.where(reason == 'Other',1,0)
        reason_reputation = np.where(reason == 'Close to Home',1,0)
        guardian_father = np.where(guardian == 'Father',1,0)
        guardian_mother = np.where(guardian == 'Mother',1,0)
        guardian_other = np.where(guardian == 'Other',1,0)
        schoolsup_no = np.where(schoolsup == 'Yes',0,1)
        schoolsup_yes = np.where(schoolsup == 'Yes',1,0)
        famsup_no = np.where(famsup == 'Yes',0,1)
        famsup_yes = np.where(famsup == 'Yes',1,0)
        paid_no = np.where(paid == 'Yes',0,1)
        paid_yes = np.where(paid == 'Yes',1,0)
        activities_no = np.where(activities == 'Yes',0,1)
        activities_yes = np.where(activities == 'Yes',1,0)
        nursery_no = np.where(nursery == 'Yes',0,1)
        nursery_yes = np.where(nursery == 'Yes',1,0)
        higher_no = np.where(higher == 'Yes',0,1)
        higher_yes = np.where(higher == 'Yes',1,0)
        internet_no = np.where(internet == 'Yes',0,1)
        internet_yes = np.where(internet == 'Yes',1,0)
        romantic_no = np.where(romantic == 'Yes',0,1)
        romantic_yes = np.where(romantic == 'Yes',1,0)

        data = {'age': age,
            'absences': absences,
             'Medu': Medu,
            'Fedu': Fedu,
            'traveltime': traveltime,
            'studytime': studytime,
            'failures': failures,
            'famrel': famrel,
            'freetime': freetime,
            'goout': goout,
            'Dalc': Dalc,
            'Walc': Walc,
            'health': health,
            'school_GP': school_GP,
            'school_MS': school_MS,
            'sex_F': sex_F,
            'sex_M': sex_M,
            'address_R': address_R,
            'address_U': address_U,
            'famsize_GT3': famsize_GT3,
            'famsize_LE3': famsize_LE3,
            'Pstatus_A': Pstatus_A,
            'Pstatus_T': Pstatus_T,
            'Mjob_at_home': Mjob_at_home,
            'Mjob_health': Mjob_health,
            'Mjob_other': Mjob_other,
            'Mjob_services': Mjob_services,
            'Mjob_teacher': Mjob_teacher,
            'Fjob_at_home': Fjob_at_home,
            'Fjob_health': Fjob_health,
            'Fjob_other': Fjob_other,
            'Fjob_services': Fjob_services,
            'Fjob_teacher': Fjob_teacher,
            'reason_course': reason_course,
            'reason_home': reason_home,
            'reason_other': reason_other,
            'reason_reputation': reason_reputation,
            'guardian_father': guardian_father,
            'guardian_mother': guardian_mother,
            'guardian_other': guardian_other,
            'schoolsup_no': schoolsup_no,
            'schoolsup_yes': schoolsup_yes,
            'famsup_no': famsup_no,
            'famsup_yes': famsup_yes,
            'paid_no': paid_no,
            'paid_yes': paid_yes,
            'activities_no': activities_no,
            'activities_yes': activities_yes,
            'nursery_no': nursery_no,
            'nursery_yes': nursery_yes,
            'famsize_LE3': famsize_LE3,
            'higher_no': higher_no,
            'higher_yes': higher_yes,
            'internet_no': internet_no,
            'internet_yes': internet_yes,
            'romantic_no': romantic_no,
            'romantic_yes': romantic_yes,
            }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()


st.subheader('User Input features')
st.write(input_df)


load_rf = pickle.load(open('por.pkl', 'rb'))

prediction = load_rf.predict(input_df)

st.subheader('Prediction')
st.write(prediction)

X = pd.read_csv('X.csv')

explainer = shap.TreeExplainer(load_rf)
shap_values = explainer.shap_values(X)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.header('Feature Importance')
plt.title('Feature Importance based on SHAP values combined')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches = 'tight')
st.write('---')

for i in np.arange(0,14):
    st.header('Feature Importance for Grade {}'.format(i+4))
    plt.title('Feature Importance based on SHAP')
    shap.summary_plot(shap_values[i], X)
    st.pyplot(bbox_inches = 'tight')
    st.write('---')


# In[6]:


#!jupyter nbconvert por.ipynb --to python


# In[ ]:




