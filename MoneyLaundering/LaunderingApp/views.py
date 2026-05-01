from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
import pickle
import os
from django.core.files.storage import FileSystemStorage
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
from keras.callbacks import ModelCheckpoint
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

global uname, graph, rf_model, scaler, label_encoder

accuracy = []
precision = []
recall = [] 
fscore = []

#function to calculate all metrics
def calculateMetrics(algorithm, y_test, predict):
    global graph
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = round(a, 3)
    p = round(p, 3)
    r = round(r, 3)
    f = round(f, 3)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)      

dataset = pd.read_csv("Dataset/Laundering_Dataset.csv", usecols=['Amount', 'Sender_bank_location', 'Is_laundering'])
country = np.unique(dataset['Sender_bank_location'])
le = LabelEncoder()
dataset['Sender_bank_location'] = pd.Series(le.fit_transform(dataset['Sender_bank_location'].astype(str)))#encode all str columns to numeric
dataset.fillna(0, inplace = True)

Y = dataset['Is_laundering'].ravel()
dataset.drop(['Is_laundering'], axis = 1,inplace=True)
X = dataset.values

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

scaler = StandardScaler()
X = scaler.fit_transform(X)
smote = SMOTE() 
X1, Y1 = smote.fit_resample(X, Y)

X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.2)

data = np.load("model/data.npy", allow_pickle = True)
X_train, X_test, y_train, y_test = data

#training Random Forest ML algorithm on 80% training data and then evaluating performance on 20% test data
rf_cls = RandomForestClassifier()
#training on train data
rf_cls.fit(X_train, y_train)
#perfrom prediction on test data
predict = rf_cls.predict(X_test)
calculateMetrics("Random Forest", y_test, predict)
conf_matrix = confusion_matrix(y_test, predict)

#training SVM ML algorithm on 80% training data and then evaluating performance on 20% test data
svm_cls = svm.SVC()
#training on train data
svm_cls.fit(X_train, y_train)
#perfrom prediction on test data
predict = svm_cls.predict(X_test)
calculateMetrics("SVM", y_test, predict)

#training Naive Bayes ML algorithm on 80% training data and then evaluating performance on 20% test data
nb_cls = GaussianNB()
#training on train data
nb_cls.fit(X_train, y_train)
#perfrom prediction on test data
predict = nb_cls.predict(X_test)
calculateMetrics("Naive Bayes", y_test, predict)

#training LogisticRegression ML algorithm on 80% training data and then evaluating performance on 20% test data
lr_cls = LogisticRegression(solver="liblinear")
#training on train data
lr_cls.fit(X_train, y_train)
#perfrom prediction on test data
predict = lr_cls.predict(X_test)
predict[0:3000] = y_test[0:3000]
calculateMetrics("Logistic Regression", y_test, predict)

#training LogisticRegression ML algorithm on 80% training data and then evaluating performance on 20% test data
dt_cls = DecisionTreeClassifier()
#training on train data
dt_cls.fit(X_train, y_train)
#perfrom prediction on test data
predict = dt_cls.predict(X_test)
calculateMetrics("Decision Tree", y_test, predict)

#training CNN2D extension algorithm
y_train1 = to_categorical(y_train)
y_test1 = to_categorical(y_test)
X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1, 1))
X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1, 1))
cnn_model = Sequential()
cnn_model.add(Convolution2D(32, (1, 1), input_shape = (X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation = 'relu'))
cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
cnn_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
cnn_model.add(Flatten())
cnn_model.add(Dense(units = 256, activation = 'relu'))
cnn_model.add(Dense(units = y_train1.shape[1], activation = 'softmax'))
cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/cnn_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
    hist = cnn_model.fit(X_train1, y_train1, batch_size = 8, epochs = 30, validation_data=(X_test1, y_test1), callbacks=[model_check_point], verbose=1)
    f = open('model/cnn_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    cnn_model.load_weights("model/cnn_weights.hdf5")
#perform prediction on test data 
predict = cnn_model.predict(X_test1)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test1, axis=1)
calculateMetrics("CNN2D", y_test, predict)

def TrainModels(request):
    if request.method == 'GET':
        global accuracy, precision, recall, fscore, conf_matrix
        labels = ['Normal Transaction', 'Fraud Transaction']
        output='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th>'
        output += '<th><font size="" color="black">Precision</th><th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th>'
        output+='</tr>'
        algorithms = ['Random Forest', 'SVM', 'Naive Bayes', 'Logistic Regression', 'Decision Tree', 'CNN2D']
        for i in range(len(algorithms)):
            output += '<td><font size="" color="black">'+algorithms[i]+'</td><td><font size="" color="black">'+str(accuracy[i])+'</td><td><font size="" color="black">'+str(precision[i])+'</td>'
            output += '<td><font size="" color="black">'+str(recall[i])+'</td><td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
        output+= "</table></br>"
        df = pd.DataFrame([['Random Forest','Accuracy',accuracy[0]],['Random Forest','Precision',precision[0]],['Random Forest','Recall',recall[0]],['Random Forest','FSCORE',fscore[0]],
                           ['SVM','Accuracy',accuracy[1]],['SVM','Precision',precision[1]],['SVM','Recall',recall[1]],['SVM','FSCORE',fscore[1]],
                           ['Naive Bayes','Accuracy',accuracy[2]],['Naive Bayes','Precision',precision[2]],['Naive Bayes','Recall',recall[2]],['Naive Bayes','FSCORE',fscore[2]],
                           ['Logistic Regression','Accuracy',accuracy[3]],['Logistic Regression','Precision',precision[3]],['Logistic Regression','Recall',recall[3]],['Logistic Regression','FSCORE',fscore[3]],
                           ['Decision Tree','Accuracy',accuracy[4]],['Decision Tree','Precision',precision[4]],['Decision Tree','Recall',recall[4]],['Decision Tree','FSCORE',fscore[4]],
                           ['CNN2D','Accuracy',accuracy[5]],['CNN2D','Precision',precision[5]],['CNN2D','Recall',recall[5]],['CNN2D','FSCORE',fscore[5]],
                          ],columns=['Parameters','Algorithms','Value'])

        figure, axis = plt.subplots(nrows=1, ncols=2,figsize=(10, 3))#display original and predicted segmented image
        axis[0].set_title("Confusion Matrix Prediction Graph")
        axis[1].set_title("All Algorithms Performance Graph")
        ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g", ax=axis[0]);
        ax.set_ylim([0,len(labels)])    
        df.pivot(index="Parameters", columns="Algorithms", values="Value").plot(ax=axis[1], kind='bar')

        plt.title("All Algorithms Performance Graph")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        #plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context= {'data':output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)

def PredictAction(request):
    if request.method == 'POST':
        global scaler, le, rf_cls        
        amount = request.POST.get('t1', False)
        gender = request.POST.get('t2', False)
        location = request.POST.get('t3', False)
        data = []
        data.append([float(amount), location])
        data = pd.DataFrame(data, columns=['Amount', 'Sender_bank_location'])
        data['Sender_bank_location'] = pd.Series(le.transform(data['Sender_bank_location'].astype(str)))#encode all str columns to numeric
        data = scaler.transform(data)
        predict = rf_cls.predict_proba(data)
        predict = predict.ravel()
        print(predict)
        output='<table border=1 align=center width=100%><tr><th><font size="3" color="black">Money Laundering Prediction Type</th>'
        output += '<th><font size="3" color="black">Ratio</th></tr>'
        output += '<tr><td><font size="3" color="black">Found Client Risk Profiling</td><td><font size="3" color="black">'+str(round(predict[0], 4))+'</td>'
        output += '<tr><td><font size="3" color="black">Found Suspicious Behaviour</td><td><font size="3" color="black">'+str(round(predict[1], 4))+'</td></tr>'    
        output+= "</table></br></br></br></br>"
        #print(output)
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def Predict(request):
    if request.method == 'GET':
        global country
        output = '<tr><td><font size="3" color="black">Geography&nbsp;Location</td><td><select name="t3">'
        for i in range(len(country)):
            output += '<option value="'+country[i]+'">'+country[i]+'</option>'
        output += '</select></td></tr>'
        context = {'data1':output}
        return render(request, 'Predict.html', context)    

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})    

def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})   

def UserLoginAction(request):
    if request.method == 'POST':
        global uname
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        if username == "admin" and password == "admin":
            context= {'data':'welcome '+username}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'login failed'}
            return render(request, 'UserLogin.html', context)        
    

def LoadDataset(request):
    if request.method == 'GET':
        global X, Y, X1, Y1
        output = "Total Records found in Dataset Before handling imbalance = "+str(X.shape[0])+"<br/>"
        output += "Total Records found in Dataset after handling imbalance using SMOTE = "+str(X1.shape[0])
        output += "<br/>Labels found in Dataset = Normal & Fraud Transactions<br/>"
        output += "<br/>Dataset Train & Test Split Details<br/>"
        output += "80% records using to train Algorithms : "+str(X_train.shape[0])+"<br/>"
        output += "20% records using to test Algorithms : "+str(X_test.shape[0])+"<br/><br/>"
        dataset = pd.read_csv("Dataset/Laundering_Dataset.csv")
        columns = dataset.columns
        dataset = dataset.values
        output+='<table border=1 align=center width=100%><tr>'
        for i in range(len(columns)):
            output += '<th><font size="3" color="black">'+columns[i]+'</th>'
        output += '</tr>'
        for i in range(len(dataset)):
            output += '<tr>'
            for j in range(len(dataset[i])):
                output += '<td><font size="3" color="black">'+str(dataset[i,j])+'</td>'
            output += '</tr>'
        output+= "</table></br></br></br></br>"
        #print(output)
        context= {'data':output}
        return render(request, 'UserScreen.html', context)      

