import streamlit as st
import pandas as pd
import numpy as np
import joblib



'''
# Подбор геометрических параметров микроструктуры
#### Чтобы прогнозировать теплоотдачу с помощью "нейросетей", используйте переключатели ниже:
'''
#nanowires
lm_nanowires_water=joblib.load('linear_regression_nanowires_water.pkl')
lm_nanowires_ethanol=joblib.load('linear_regression_nanowires_ethanol.pkl')
lm_nanowires_FC72=joblib.load('linear_regression_nanowires_FC-72.pkl')
lm_nanowires_PF5060=joblib.load('linear_regression_nanowires_PF-5060.pkl')
ridgemodel_nanowires_water=joblib.load('ridge_nanowires_water.pkl')
ridgemodel_nanowires_ethanol=joblib.load('ridge_nanowires_ethanol.pkl')
ridgemodel_nanowires_FC72=joblib.load('ridge_nanowires_FC-72.pkl')
ridgemodel_nanowires_PF5060=joblib.load('ridge_nanowires_PF-5060.pkl')
dectreemodel_nanowires_water=joblib.load('decision_tree_nanowires_water.pkl')
dectreemodel_nanowires_ethanol=joblib.load('decision_tree_nanowires_ethanol.pkl')
dectreemodel_nanowires_FC72=joblib.load('decision_tree_nanowires_FC-72.pkl')
dectreemodel_nanowires_PF5060=joblib.load('decision_tree_nanowires_PF-5060.pkl')
rndmforestmodel_nanowires_water=joblib.load('random_forest_nanowires_water.pkl')
rndmforestmodel_nanowires_ethanol=joblib.load('random_forest_nanowires_ethanol.pkl')
rndmforestmodel_nanowires_FC72=joblib.load('random_forest_nanowires_FC-72.pkl')
rndmforestmodel_nanowires_PF5060=joblib.load('random_forest_nanowires_PF-5060.pkl')

#nanostructures
lm_nanostructures_water=joblib.load('linear_regression_nanostructures_water.pkl')
lm_nanostructures_ethanol=joblib.load('linear_regression_nanostructures_ethanol.pkl')
lm_nanostructures_FC72=joblib.load('linear_regression_nanostructures_FC-72.pkl')
ridgemodel_nanostructures_water=joblib.load('ridge_nanostructures_water.pkl')
ridgemodel_nanostructures_ethanol=joblib.load('ridge_nanostructures_ethanol.pkl')
ridgemodel_nanostructures_FC72=joblib.load('ridge_nanostructures_FC-72.pkl')
dectreemodel_nanostructures_water=joblib.load('decision_tree_nanostructures_water.pkl')
dectreemodel_nanostructures_ethanol=joblib.load('decision_tree_nanostructures_ethanol.pkl')
dectreemodel_nanostructures_FC72=joblib.load('decision_tree_nanostructures_FC-72.pkl')
rndmforestmodel_nanostructures_water=joblib.load('random_forest_nanostructures_water.pkl')
rndmforestmodel_nanostructures_ethanol=joblib.load('random_forest_nanostructures_ethanol.pkl')
rndmforestmodel_nanostructures_FC72=joblib.load('random_forest_nanostructures_FC-72.pkl')

#porous fins
lm_porous_fins_water=joblib.load('linear_regression_porous-fins_water.pkl')
lm_porous_fins_n_Pentane=joblib.load('linear_regression_porous-fins_n-Pentane.pkl')
ridgemodel_porous_fins_water=joblib.load('ridge_porous-fins_water.pkl')
ridgemodel_porous_fins_n_Pentane=joblib.load('ridge_porous-fins_n-Pentane.pkl')
dectreemodel_porous_fins_water=joblib.load('decision_tree_porous-fins_water.pkl')
dectreemodel_porous_fins_n_Pentane=joblib.load('decision_tree_porous-fins_n-Pentane.pkl')
rndmforestmodel_porous_fins_water=joblib.load('random_forest_porous-fins_water.pkl')
rndmforestmodel_porous_fins_n_Pentane=joblib.load('random_forest_porous-fins_n-Pentane.pkl')

colum1, colum2, colum3, colum4= st.columns(4)
with colum1:
    linReg=st.checkbox("LinearRegression", False)
with colum2:
    ridge=st.checkbox("Ridge", False)
with colum3:
    decisiontree=st.checkbox("DecisionTree", False)
with colum4:
    randomforest=st.checkbox("RandomForest", False)

genre = st.radio("Выберите вид структуры: Щеточно-волокнистая, наноструктура или пористые ребра",('щеточно-волокнистая', 'наноструктура', 'пористые ребра'))

if genre == 'щеточно-волокнистая':
    liquid = st.sidebar.selectbox('Выберите жидкость', ('Вода','Этанол','FC-72','PF-5060'))

    col1, col2= st.columns(2)
    with col1:
        st.header("Щеточно-волокнистая")
        st.image('щеточно-волокнистые.jpg',  use_column_width=True)

    if liquid=='Вода':
       #q 124 index
        x1 = st.sidebar.slider('q (в Вт)', min_value=20000, max_value=1670000,  value=300000)
       #h
        x3 = st.sidebar.slider('Высота волокна (h в нм)', min_value=450, max_value=32000,  value=5000)
       #d
        x5 = st.sidebar.slider('Толщина / ширина волокна (δ/s в нм)', min_value=40, max_value=850,  value=400)
       
        data_slider = {'h': [x3/1000], 'delta thickness': [x5/1000], 's width': [x5/1000], 'q': [x1]}
        nm = pd.DataFrame(data=data_slider)
        with col2:
            st.header("Значение интесификации теплоотдачи")  
            st.write('q =', round(x1/1000, 1),'кВт; ', 'h =', x3,'нм; ', 'δ/s =', x5,'нм')
            if linReg:
                y_linReg = lm_nanowires_water.predict(nm)
                st.write('LinearRegression: α/α0=',round(y_linReg[0], 2))
            if ridge:
                y_ridge = ridgemodel_nanowires_water.predict(nm)
                st.write('Ridge: α/α0=',round(y_ridge[0], 2))
            if decisiontree:
                y_decisiontree=dectreemodel_nanowires_water.predict(nm)
                st.write('DecisionTree: α/α0=',round(y_decisiontree[0], 2))
            if randomforest:
                y_randomforest=rndmforestmodel_nanowires_water.predict(nm)
                st.write('RandomForest: α/α0=',round(y_forest[0], 2))
        
    elif liquid=='Этанол':
       #q 124 index
        x1 = st.sidebar.slider('q (в Вт)', min_value=35000, max_value=640000,  value=250000)
       #h
        x3 = st.sidebar.slider('Высота волокна (h в нм)', min_value=1300, max_value=1300,  value=1300)
       #d
        x5 = st.sidebar.slider('Толщина / ширина волокна (δ/s в нм)', min_value=550, max_value=550,  value=550)
        
        data_slider = {'h': [x3/1000], 'delta thickness': [x5/1000], 's width': [x5/1000], 'q': [x1]}
        nm = pd.DataFrame(data=data_slider)
        with col2:
            st.header("Значение интесификации теплоотдачи")  
            st.write('q =', round(x1/1000, 1),'кВт; ', 'h =', x3,'нм; ', 'δ/s =', x5,'нм')
            if linReg:
                y_linReg = lm_nanowires_ethanol.predict(nm)
                st.write('LinearRegression: α/α0=',round(y_linReg[0], 2))
            if ridge:
                y_ridge = ridgemodel_nanowires_ethanol.predict(nm)
                st.write('Ridge: α/α0=',round(y_ridge[0], 2))
            if decisiontree:
                y_decisiontree=dectreemodel_nanowires_ethanol.predict(nm)
                st.write('DecisionTree: α/α0=',round(y_decisiontree[0], 2))
            if randomforest:
                y_randomforest=rndmforestmodel_nanowires_ethanol.predict(nm)
                st.write('RandomForest: α/α0=',round(y_forest[0], 2))
        
    elif liquid=='FC-72':
       #q 124 index
        x1 = st.sidebar.slider('q (в Вт)', min_value=1000, max_value=150000,  value=50000)
       #h
        x3 = st.sidebar.slider('Высота волокна (h в нм)', min_value=155, max_value=644,  value=300)
       #d
        x5 = st.sidebar.slider('Толщина / ширина волокна (δ/s в нм)', min_value=100, max_value=270,  value=180)
        
        data_slider = {'h': [x3/1000], 'delta thickness': [x5/1000], 's width': [x5/1000], 'q': [x1]}
        nm = pd.DataFrame(data=data_slider)
        with col2:
            st.header("Значение интесификации теплоотдачи")  
            st.write('q =', round(x1/1000, 1),'кВт; ', 'h =', x3,'нм; ', 'δ/s =', x5,'нм')
            if linReg:
                y_linReg = lm_nanowires_FC72.predict(nm)
                st.write('LinearRegression: α/α0=',round(y_linReg[0], 2))
            if ridge:
                y_ridge = ridgemodel_nanowires_FC72.predict(nm)
                st.write('Ridge: α/α0=',round(y_ridge[0], 2))
            if decisiontree:
                y_decisiontree=dectreemodel_nanowires_FC72.predict(nm)
                st.write('DecisionTree: α/α0=',round(y_decisiontree[0], 2))
            if randomforest:
                y_randomforest=rndmforestmodel_nanowires_FC72.predict(nm)
                st.write('RandomForest: α/α0=',round(y_forest[0], 2))   
                
    else:
       #q 124 index
        x1 = st.sidebar.slider('q (в Вт)', min_value=3000, max_value=54000,  value=25000)
       #h
        x3 = st.sidebar.slider('Высота волокна (h в нм)', min_value=9, max_value=25,  value=15)
       #d
        x5 = st.sidebar.slider('Толщина / ширина волокна (δ/s в нм)', min_value=13, max_value=13,  value=13)
        
        data_slider = {'h': [x3/1000], 'delta thickness': [x5/1000], 's width': [x5/1000], 'q': [x1]}
        nm = pd.DataFrame(data=data_slider)
        with col2:
            st.header("Значение интесификации теплоотдачи")  
            st.write('q =', round(x1/1000, 1),'кВт; ', 'h =', x3,'нм; ', 'δ/s =', x5,'нм')
            if linReg:
                y_linReg = lm_nanowires_PF5060.predict(nm)
                st.write('LinearRegression: α/α0=',round(y_linReg[0], 2))
            if ridge:
                y_ridge = ridgemodel_nanowires_PF5060.predict(nm)
                st.write('Ridge: α/α0=',round(y_ridge[0], 2))
            if decisiontree:
                y_decisiontree=dectreemodel_nanowires_PF5060.predict(nm)
                st.write('DecisionTree: α/α0=',round(y_decisiontree[0], 2))
            if randomforest:
                y_randomforest=rndmforestmodel_nanowires_PF5060.predict(nm)
                st.write('RandomForest: α/α0=',round(y_forest[0], 2))



elif genre == 'наноструктура':
    liquid = st.sidebar.selectbox('Выберите жидкость', ('Вода','Этанол','FC-72'))

    if liquid=='Вода':
       #q 124 index
        x1 = st.sidebar.slider('q (в Вт)', min_value=30000, max_value=2190000,  value=300000)
       #h
        x3 = st.sidebar.slider('Высота структуры (h в нм)', min_value=10000, max_value=40000,  value=25000)
       #d
        x5 = st.sidebar.slider('Толщина / ширина структуры (δ/s в нм)', min_value=5000, max_value=20000,  value=10000)
       #u
        x7 = st.sidebar.slider('Шаг между элементами структуры (Δ/u в нм)', min_value=5000, max_value=40000,  value=10000)
    elif liquid=='Этанол':
       #q 124 index
        x1 = st.sidebar.slider('q (в Вт)', min_value=10000, max_value=640000,  value=50000)
       #h
        x3 = st.sidebar.slider('Высота структуры (h в нм)', min_value=900, max_value=50000,  value=10000)
       #d
        x5 = st.sidebar.slider('Толщина / ширина структуры (δ/s в нм)', min_value=400, max_value=50000,  value=10000)
       #u
        x7 = st.sidebar.slider('Шаг между элементами структуры (Δ/u в нм)', min_value=750, max_value=50000,  value=10000)
    else:
       #q 124 index
        x1 = st.sidebar.slider('q (в Вт)', min_value=10000, max_value=600000,  value=30000)
       #h
        x3 = st.sidebar.slider('Высота структуры (h в нм)', min_value=60000, max_value=60000,  value=60000)
       #d
        x5 = st.sidebar.slider('Толщина / ширина структуры (δ/s в нм)', min_value=30000, max_value=30000,  value=30000)
       #u
        x7 = st.sidebar.slider('Шаг между элементами структуры (Δ/u в нм)', min_value=60000, max_value=60000,  value=60000)
   

    data_slider = {'h': [x3/1000], 'delta thickness': [x5/1000], 's width': [x5/1000], 'DeltaCap longitudinal pitch': [x7/1000], 'u transverse pitch': [x7/1000], 'q': [x1]}
    nm = pd.DataFrame(data=data_slider)
    
    col1, col2= st.columns(2)
    with col1:
        st.header("Наноструктура")
        st.image('наноструктуры.jpg',  use_column_width=True)
    with col2:
        st.header("Значение интесификации теплоотдачи")  
        st.write('q =', round(x1/1000, 1),'кВт; ', 'h =', x3,'нм; ', 'δ/s =', x5,'нм; ', 'Δ/u =', x7, 'нм')
        if linReg:
            y_linReg = lm.predict(nm)
            st.write('LinearRegression: α/α0=',round(y_linReg[0], 2))
        if ridge:
            y_ridge = ridgemodel.predict(nm)
            st.write('Ridge: α/α0=',round(y_ridge[0], 2))
        if decisiontree:
            y_decisiontree=dectreemodel.predict(nm)
            st.write('DecisionTree: α/α0=',round(y_decisiontree[0], 2))
        if randomforest:
            y_randomforest=rndmforestmodel.predict(nm)
            st.write('RandomForest: α/α0=',round(y_forest[0], 2))

else:
    liquid = st.sidebar.selectbox('Выберите жидкость', ('Вода','n-Pentane'))

    if liquid=='Вода':
       #q 124 index
        x1 = st.sidebar.slider('q (в Вт)', min_value=38000, max_value=1199000,  value=300000)
       #h
        x3 = st.sidebar.slider('Высота структуры (h в нм)', min_value=1100000, max_value=2000000,  value=1500000)
       #d
        x5 = st.sidebar.slider('Толщина / ширина структуры (δ/s в нм)', min_value=800000, max_value=1000000,  value=900000)
       #u
        x7 = st.sidebar.slider('Шаг между элементами структуры (Δ/u в нм)', min_value=400000, max_value=1000000,  value=700000)
       #porosity
        x9 = st.sidebar.slider('Пористость, %', min_value=33, max_value=58,  value=40)
    else:
       #q 124 index
        x1 = st.sidebar.slider('q (в Вт)', min_value=50000, max_value=815000,  value=100000)
       #h
        x3 = st.sidebar.slider('Высота структуры (h в нм)', min_value=710000, max_value=2000000,  value=1000000)
       #d
        x5 = st.sidebar.slider('Толщина / ширина структуры (δ/s в нм)', min_value=530000, max_value=1800000,  value=1000000)
       #u
        x7 = st.sidebar.slider('Шаг между элементами структуры (Δ/u в нм)', min_value=830000, max_value=1900000,  value=1000000)
       #porosity
        x9 = st.sidebar.slider('Пористость, %', min_value=22, max_value=40,  value=30)

   

    data_slider = {'h': [x3/1000], 'delta thickness': [x5/1000], 's width': [x5/1000], 'DeltaCap longitudinal pitch': [x7/1000], 'u transverse pitch': [x7/1000], 'porosity, %': [x9], 'q': [x1]}
    nm = pd.DataFrame(data=data_slider)
    
    col1, col2= st.columns(2)
    with col1:
        st.header("Пористые ребра")
        st.image('пористые ребра.jpg',  use_column_width=True)
    with col2:
        st.header("Значение интесификации теплоотдачи")  
        st.write('q =', round(x1/1000, 1),'кВт; ', 'h =', x3,'нм; ', 'δ/s =', x5,'нм; ', 'Δ/u =', x7, 'нм', 'пористость =', x9)
        if linReg:
            y_linReg = lm.predict(nm)
            st.write('LinearRegression: α/α0=',round(y_linReg[0], 2))
        if ridge:
            y_ridge = ridgemodel.predict(nm)
            st.write('Ridge: α/α0=',round(y_ridge[0], 2))
        if decisiontree:
            y_decisiontree=dectreemodel.predict(nm)
            st.write('DecisionTree: α/α0=',round(y_decisiontree[0], 2))
        if randomforest:
            y_randomforest=rndmforestmodel.predict(nm)
            st.write('RandomForest: α/α0=',round(y_forest[0], 2))





