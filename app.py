import streamlit as st
import pandas as pd
import numpy as np
import joblib



'''
# Подбор геометрических параметров микроструктуры
#### Чтобы прогнозировать теплоотдачу с помощью "нейросетей", используйте переключатели ниже:
'''

rndm = joblib.load('rndmF_model.pkl')
Xmodel=joblib.load('XGBR_model.pkl')
lm=joblib.load('GBR_model.pkl')

rndm2 = joblib.load('rndmF_model_2.pkl')
Xmodel2=joblib.load('XGBR_model_2.pkl')
lm2=joblib.load('GBR_model_2.pkl')


colum1, colum2, colum3= st.beta_columns(3)
with colum1:
    rndFors=st.checkbox("RandomForest", False)
with colum2:
    linReg=st.checkbox("GBRegressor", False)
with colum3:
    nerKa=st.checkbox("XGBoost", False)

l0 = st.sidebar.selectbox('Выберите жидкость', ('Вода','Этанол','60% глицерин','Фреоны'))

genre = st.radio("Выберите вид структуры: 3D или 2D",('3D', '2D'))
if genre == '3D':
#q 124 index
    x1 = st.sidebar.slider('q (в Вт)', min_value=2400, max_value=2500000,  value=189327)

#угол/90
    x2 = st.sidebar.slider('Угол наклона структуры', min_value=70, max_value=90,  value=80)

#h/lo
    x3 = st.sidebar.slider('Высота ребера (h в мкм)', min_value=220, max_value=750,  value=570)

#D/lo
    x4 = st.sidebar.slider('Продольный шаг ребера (Δ в мкм)', min_value=5, max_value=350,  value=210)

#d/lo
    x5 = st.sidebar.slider('Толщина ребра (δ в мкм)', min_value=105, max_value=365,  value=140)

#u/lo
    x6 = st.sidebar.slider('Поперечный шаг ребра (u в мкм)', min_value=10, max_value=300,  value=180)

#s/lo
    x7 = st.sidebar.slider('Ширина ребра (s в мкм)', min_value=50, max_value=830,  value=140)
#Pr
    x8 = st.sidebar.slider('Pr', min_value=1.7, max_value=6.8,  value=1.75)
    
    if l0=='Вода':
        l_g=0.002502867
        r=2256800
        p_par=0.598
        u=0.000000295
        Kq_bezq=l_g/r/p_par/u
    elif l0=='Этанол':
        l_g=0.001437944
        r=800000
        p_par=1.85
        u=0.0000005366
        Kq_bezq=l_g/r/p_par/u
    elif l0=='60% глицерин':
        l_g=0.003112722
        r=2520000
        p_par=1.104
        u=0.00000051
        Kq_bezq=l_g/r/p_par/u
    else:
        l_g=0.001038336
        r=146120
        p_par=5.87
        u=0.0000002875
        Kq_bezq=l_g/r/p_par/u

    y=1.49*(x1*Kq_bezq)**(-0.15)*(x2/90)**(-1.720)*(x3/1000000/l_g)**(0.313)*(x4/1000000/l_g)**(0.069)*(x5/1000000/l_g)**(0.078)*(x6/1000000/l_g)**(-0.454)*(x7/1000000/l_g)**(-0.492)   
    data_slider = {'Kq': [x1*Kq_bezq], 'angle/90': [x2/90], 'h/lo': [x3/1000000/l_g], 'D/lo': [x4/1000000/l_g], 'd/lo': [x5/1000000/l_g], 'u/lo': [x6/1000000/l_g], 's/lo': [x7/1000000/l_g], 'Pr': [x8]}
    nm = pd.DataFrame(data=data_slider)
    
    col1, col2= st.beta_columns(2)
    with col1:
        st.header("3D структура")
        st.image('3d.jpg',  use_column_width=True)
    with col2:
        st.header("Значение интесификации теплоотдачи")  
        st.write('q=', round(x1/1000, 1),'кВт; ','угол=', x2,'°; ','h=', x3,'мкм; ','Δ=', x4,'мкм; ','δ=', x5,'мкм; ','u=', x6,'мкм; ','s=', x7, 'мкм; ','Pr', x8, 'мкм')
        st.write('Полиноминальная регрессия: α/α0=',round(y, 2))
        if rndFors:
            y_forest=rndm.predict(nm)
            st.write('RandomForest: α/α0=',round(y_forest[0], 2))
        if linReg:
            y_linReg = lm.predict(nm)
            st.write('GBRegressor: α/α0=',round(y_linReg[0], 2))
        if nerKa:
            y_nerKa = Xmodel.predict(nm)  #(xnm)
            st.write('XGBoost: α/α0=',round(y_nerKa[0], 2))
else:
#q 676 index
    x1 = st.sidebar.slider('q (в Вт)', min_value=3800, max_value=2200000,  value=50283)

#угол/90
    x2 = st.sidebar.slider('Угол наклона структуры', min_value=65, max_value=90,  value=90)

#h/lo
    x3 = st.sidebar.slider('Высота ребера (h в мкм)', min_value=90, max_value=1530,  value=1038)

#D/lo
    x4 = st.sidebar.slider('Продольный шаг ребера (Δ в мкм)', min_value=5, max_value=1360,  value=450)

#d/lo
    x5 = st.sidebar.slider('Толщина ребра (δ в мкм)', min_value=25, max_value=1050,  value=1050)
#Pr
    x6 = st.sidebar.slider('Pr', min_value=1.7, max_value=6.8,  value=1.75)
    if l0=='Вода':
        l_g=0.002502867
        r=2256800
        p_par=0.598
        u=0.000000295
        Kq_bezq=l_g/r/p_par/u
    elif l0=='Этанол':
        l_g=0.001437944
        r=800000
        p_par=1.85
        u=0.0000005366
        Kq_bezq=l_g/r/p_par/u
    elif l0=='60% глицерин':
        l_g=0.003112722
        r=2520000
        p_par=1.104
        u=0.00000051
        Kq_bezq=l_g/r/p_par/u
    else:
        l_g=0.001038336
        r=146120
        p_par=5.87
        u=0.0000002875
        Kq_bezq=l_g/r/p_par/u
        
    y=2.66*(x1*Kq_bezq)**(-0.09)*(x2/90)**(-0.091)*(x3/1000000/l_g)**(0.133)*(x4/1000000/l_g)**(0.035)*(x5/1000000/l_g)**(-0.149)
    
    data_slider = {'Kq': [x1*Kq_bezq], 'angle/90': [x2/90], 'h/lo': [x3/1000000/l_g], 'D/lo': [x4/1000000/l_g], 'd/lo': [x5/1000000/l_g], 'Pr': [x6]}
    nm = pd.DataFrame(data=data_slider)
    
    col1, col2= st.beta_columns(2)
    with col1:
        st.header("2D структура")
        st.image('2d.jpg',  use_column_width=True)
    with col2:
        st.header("Значение интенсификации теплоотдачи")
        st.write('q=', round(x1/1000, 1),'кВт; ','угол=', x2,'°; ','h=', x3,'мкм; ','Δ=', x4,'мкм; ','δ=', x5,'мкм; ','Pr', x6, 'мкм')
        st.write('Полиноминальная регрессия: α/α0=',round(y, 2))
        if rndFors:
            y_forest=rndm2.predict(nm)
            st.write('RandomForest: α/α0=',round(y_forest[0], 2))
        if linReg:
            y_linReg = lm2.predict(nm)
            st.write('GBRegressor: α/α0=',round(y_linReg[0], 2))
        if nerKa:
            y_nerKa = Xmodel2.predict(nm)
            st.write('XGBoost: α/α0=',round(y_nerKa[0], 2))
