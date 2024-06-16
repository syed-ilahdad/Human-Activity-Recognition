from turtle import title
from django.shortcuts import render, HttpResponse
from .forms import UserRegistrationForm
from django.contrib import messages
from .models import UserRegistrationModel

import matplotlib.pyplot as plt
    
from django.conf import settings
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
import scipy.stats as stats
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten, Dense, Dropout



# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account has not been activated by Admin.')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})

def Eda(request):
    data = settings.MEDIA_ROOT+ "//" +'HumanActionsRaw.txt'
    file = open(data)
    lines = file.readlines()

    processedList = []

    for i, line in enumerate(lines):
        try:
            line = line.split(',')
            last = line[5].split(';')[0]
            last = last.strip()
            if last == '':
                break;
            temp = [line[0], line[1], line[2], line[3], line[4], last]
            processedList.append(temp)
        except:
            print('Error at line number: ', i)
    columns = ['user', 'activity', 'time', 'x-axis', 'y-axis', 'z-axis']
    data = pd.DataFrame(data = processedList, columns = columns)
    data['x-axis'] = data['x-axis'].astype('float')
    data['y-axis'] = data['y-axis'].astype('float')
    data['z-axis'] = data['z-axis'].astype('float')
    
    data['activity'].value_counts().plot(kind='bar', title='Number of Examples by Activities',color=['b','r','g','y','k','r']); 
    # fig_save_path = settings.MEDIA_ROOT+ "//" +'plot.png'
    plt.savefig('assets/static/plot.png')
    plt.show()
    def plot_activity(activity, df,title):
        data = df[df['activity'] == activity][['x-axis', 'y-axis', 'z-axis']][:200]
        axis = data["x-axis"].plot(subplots=True, 
                        title=title,color="b")
        axis = data["y-axis"].plot(subplots=True, 
                    title=title,color="r")
        axis = data["z-axis"].plot(subplots=True, 
                title=title,color="g")
        for ax in axis:
            ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))

    plot_activity("Sitting", data,'Tri-axial data for a single class (Sitting)')
    plt.show()
    plot_activity("Standing", data,'Tri-axial data for a single class (Standing)')
    plt.show()
    df = data.drop(['user', 'time'], axis = 1).copy()
    df.to_csv(r'media\processed_data.csv')

    return render(request, 'users/activityplot.html')

def Training(request):
    

    data = settings.MEDIA_ROOT+ "//" +'processed_data.csv'
    df = pd.read_csv(data)
    Walking = df[df['activity']=='Walking'].head(3555).copy()
    Jogging = df[df['activity']=='Jogging'].head(3555).copy()
    Upstairs = df[df['activity']=='Upstairs'].head(3555).copy()
    Downstairs = df[df['activity']=='Downstairs'].head(3555).copy()
    Sitting = df[df['activity']=='Sitting'].head(3555).copy()
    Standing = df[df['activity']=='Standing'].copy()
    balanced_data = pd.DataFrame()
    balanced_data = balanced_data.append([Walking, Jogging, Upstairs, Downstairs, Sitting, Standing])
    label = LabelEncoder()
    balanced_data['label'] = label.fit_transform(balanced_data['activity'])
    X = balanced_data[['x-axis', 'y-axis', 'z-axis']]
    y = balanced_data['label']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    scaled_X = pd.DataFrame(data = X, columns = ['x', 'y', 'z'])
    scaled_X['label'] = y.values

    Fs = 20
    frame_size = Fs*4 # 80
    hop_size = Fs*2 # 40
    def get_frames(df, frame_size, hop_size):

        N_FEATURES = 3

        frames = []
        labels = []
        for i in range(0, len(df) - frame_size, hop_size):
            x = df['x'].values[i: i + frame_size]
            y = df['y'].values[i: i + frame_size]
            z = df['z'].values[i: i + frame_size]
            
            # Retrieve the most often used label in this segment
            label = stats.mode(df['label'][i: i + frame_size])[0][0]
            frames.append([x, y, z])
            labels.append(label)

        # Bring the segments into a better shape
        frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
        labels = np.asarray(labels)

        return frames, labels

    X, y = get_frames(scaled_X, frame_size, hop_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
    X_train = X_train.reshape(425, 80, 3, 1)
    X_test = X_test.reshape(107, 80, 3, 1)
    print('X_train[0].shape:',X_train[0].shape)

    model = Sequential()
    model.add(Conv2D(16, (2, 2), activation = 'relu', input_shape = X_train[0].shape))
    model.add(Dropout(0.1))

    model.add(Conv2D(32, (2, 2), activation='relu'))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.5))

    model.add(Dense(6, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    history = model.fit(X_train, y_train, epochs = 10, validation_data= (X_test, y_test), verbose=1)

    def plot_learningCurve(history, epochs):
        # Plot training & validation accuracy values
        epoch_range = range(1, epochs+1)
        plt.plot(epoch_range, history.history['accuracy'])
        plt.plot(epoch_range, history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(epoch_range, history.history['loss'])
        plt.plot(epoch_range, history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()
    plot_learningCurve(history, 10)
    y_pred = model.predict(X_test)
    print(y_pred)
    acc = history.history['accuracy'][-1]*100
    loss = history.history['loss'][-1]*100
    model.save('media\model.h5')

    return render(request, 'users/training.html',{'acc':acc,'loss':loss})
    
 
def Predict(request):
    if request.method == 'POST':
        index_no = int(request.POST.get('index_no'))
        data = settings.MEDIA_ROOT+ "//" +'processed_data.csv'
        df = pd.read_csv(data)
        Walking = df[df['activity']=='Walking'].head(3555).copy()
        Jogging = df[df['activity']=='Jogging'].head(3555).copy()
        Upstairs = df[df['activity']=='Upstairs'].head(3555).copy()
        Downstairs = df[df['activity']=='Downstairs'].head(3555).copy()
        Sitting = df[df['activity']=='Sitting'].head(3555).copy()
        Standing = df[df['activity']=='Standing'].copy()
        balanced_data = pd.DataFrame()
        balanced_data = balanced_data.append([Walking, Jogging, Upstairs, Downstairs, Sitting, Standing])
        label = LabelEncoder()
        balanced_data['label'] = label.fit_transform(balanced_data['activity'])
        X = balanced_data[['x-axis', 'y-axis', 'z-axis']]
        y = balanced_data['label']
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        scaled_X = pd.DataFrame(data = X, columns = ['x', 'y', 'z'])
        scaled_X['label'] = y.values

        Fs = 20
        frame_size = Fs*4 # 80
        hop_size = Fs*2 # 40
        def get_frames(df, frame_size, hop_size):

            N_FEATURES = 3

            frames = []
            labels = []
            for i in range(0, len(df) - frame_size, hop_size):
                x = df['x'].values[i: i + frame_size]
                y = df['y'].values[i: i + frame_size]
                z = df['z'].values[i: i + frame_size]
                
                # Retrieve the most often used label in this segment
                label = stats.mode(df['label'][i: i + frame_size])[0][0]
                frames.append([x, y, z])
                labels.append(label)

            # Bring the segments into a better shape
            frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
            labels = np.asarray(labels)

            return frames, labels

        X, y = get_frames(scaled_X, frame_size, hop_size)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
        X_train = X_train.reshape(425, 80, 3, 1)
        X_test = X_test.reshape(107, 80, 3, 1)
        test = X_test[index_no].tolist()
        from keras.models import load_model
        model = load_model('media/model.h5')
        res = model.predict([test])
        res = res.argmax()
        activities = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']
        activity = activities[res]
        print(activity)
        return render(request,'users/Predict.html',{'activity':activity} )
    else:
        return render(request,'users/Predict.html' )
