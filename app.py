import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import seaborn as sns


st.title('Streamlit with Machine Learning')

st.write("""
# Explore different classifier and datasets
Which one is the best?
""")

dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris','Breast Cancer','Wine')
)
st.write(f" ## {dataset_name} Dataset")
st.write('To know more about dataset search them in Kaggle.')
classifier_name = st.sidebar.selectbox(
    'Select Classifier',
    ('KNN','SVM','Random Forest')
)

def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    Y = data.target
    return X,Y

X,Y = get_dataset(dataset_name)
X = pd.DataFrame(X)
Y = pd.DataFrame(Y)
con = pd.concat([X,Y],axis=1)
st.write(f'Top 5 Entries of {dataset_name} dataset', con.head())
st.write('Shape of Dataset:', X.shape)
st.write('Number of Classes:', len(np.unique(Y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=100)
    return clf

clf = get_classifier(classifier_name,params)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=100)
clf.fit(X_train,Y_train)
pred = clf.predict(X_test)
acc = accuracy_score(Y_test,pred)
cm = confusion_matrix(Y_test,pred)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy=',acc)
st.write('Confusion Matrix')
st.write(cm)
st.subheader('Heatmap of Comfusion Matrix')
plt.figure(figsize=(7,2,))
plt.imshow(cm, cmap='hot', interpolation='nearest')
st.pyplot()

st.subheader('Plot Dataset')
pca = PCA(2)
X_pca = pca.fit_transform(X)

x1 = X_pca[:,0]
x2 = X_pca[:,1]

plt.scatter(x1,x2,c=Y,alpha=0.8,cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
st.pyplot()

st.subheader('Some Histograms:')
X.hist(bins=50,figsize=(10,15))
st.pyplot()

st.subheader('Some BoxPlot:')
X.plot(kind='box',figsize=(10,15))
st.pyplot()

st.header("Yay! If you like this website then Click below to celebrate.")
btn = st.button("Celebrate!")
if btn:
    st.balloons()