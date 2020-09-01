# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 12:54:01 2018

@author: Noufanpmc
"""

#!/usr/bin/python

#==============================IMPORTING LIBRARIES============================#
from keras.preprocessing import image
from keras.preprocessing.image import load_img #method to load images
from keras.preprocessing.image import img_to_array #method to convert image to array
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import numpy as np
import time 
import os
import tensorflow as tf
import numpy as np
import keras.models
from keras.models import model_from_json
import json
from scipy.misc import imread, imresize,imshow
#requests are objects that flask handles (get set post, etc)
from flask import Flask, jsonify, redirect, render_template,request,Response
import cv2 
from sklearn.preprocessing import LabelBinarizer
import webbrowser
 
#=============================================================================#

#initalize our flask app
app = Flask(__name__)
#label=[]
#global label

#================================CAPTURING====================================#
@app.route('/', methods=['GET', 'POST'])
def index():
    for key in request.form:
        if key.startswith('comment.'):
            id_ = key.partition('.')[-1]
            value = request.form[key]
            print (id_)
            print(value)
            
            video=cv2.VideoCapture(0)
            face_cascade=cv2.CascadeClassifier("C:\\Users\\Noufanpmc\\Documents\\Aegis\\Capstone\\Project\\code\\haarcascade_frontalface_default.xml")
            os.chdir("C:\\Users\\Noufanpmc\\Documents\\Aegis\\Capstone\\Project\\code\\Capture")
            os.makedirs(str(value))
            dynamic=("C:\\Users\\Noufanpmc\\Documents\\Aegis\\Capstone\\Project\\code\\Capture\\"+str(value))
            os.chdir(dynamic)
            sampleNum=0;
            a=0
            while True:
                a=a+1
                check, frame=video.read()
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                faces=face_cascade.detectMultiScale(frame,scaleFactor=1.30,minNeighbors=5)
                print(faces)                         
                print(check)
                print(frame)
                for x,y,w,h in faces:
                    sampleNum=sampleNum+1;
                    color_img=frame[y:y+h,x:x+w]
                    resized_color_image=cv2.resize(color_img,(224,224))
                    cv2.imwrite(str(value)+"."+str(sampleNum)+".jpg",resized_color_image)
                    img=cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0), 3)
                    cv2.waitKey(1)
                cv2.imshow("Capturing image/video",frame)

                cv2.waitKey(1) # 10 FRAMES IN 1 CLASSES
                if (sampleNum>200):
                    break;

            print(a)
            video.release()
            cv2.destroyAllWindows()
            
            
            
    return render_template("index.html")
    return id_
    return value

#=============================================================================#



#==================================Training===================================#

@app.route('/train/',methods=['POST','GET'])
def train():
    global model
#    model = VGGFace()
#    print(model.summary())
#    global num_classes
#    #creating image list
#    data_path = 'C:\\Users\\Dhrubo\\Desktop\\Capture' # give the data set location
#    data = os.listdir(data_path)
#    labels=[]#to append the labels
#    img_data_list = []#con
#    num_classes = 0    
#    for dataset in data:
#        num_classes = num_classes + 1
#        img_list = os.listdir(data_path+'/'+dataset)
#        print('loaded images'+ '{}\n'.format(dataset))
#        for i in img_list:
#            ID=int(i.split('.')[0])
#            labels.append(ID)
#        for img in img_list:
#            img_path = data_path+'/'+dataset+'/'+img
#            img = image.load_img(img_path, target_size=(224,224))       
#            x = image.img_to_array(img)
#            x = np.expand_dims(x, axis = 0)
#            x = preprocess_input(x) #using mean of rgb code and subtracting it from pixels
#            print('Input image shape', x.shape)
#            img_data_list.append(x)
#            
#    #converting list into arrays
#    img_data = np.array(img_data_list)
#    print(img_data.shape)
#    img_data = np.rollaxis(img_data,1,0)
#    print(img_data.shape)
#    img_data = img_data[0]
#    print(img_data.shape)
#    
#    #One hot encoding
#    lb=LabelBinarizer()
#    Y=lb.fit_transform(labels)
#    #Y = np_utils.to_categorical(labels,num_classes)
#    
#    #shuffling the dataset(randomly arranging images for training model)
#    x,y = shuffle(img_data,Y, random_state=2)
#    
#    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 2)
#    
#    print(model.summary())
#    last_layer = model.get_layer('fc8').output
#    out = Dense(num_classes, activation = 'softmax', name = 'output')(last_layer)
#    custom_vgg_model = Model(model.input,out)
#    print(custom_vgg_model.summary())
#
#    
#    for layer in custom_vgg_model.layers[:-1]:
#        layer.trainable = False
#            
#    custom_vgg_model.compile(loss='categorical_crossentropy',optimizer = 'rmsprop',metrics = ['accuracy'])
#    
#    t = time.time()
#    hist = custom_vgg_model.fit(X_train,y_train, batch_size = 32, epochs = 10, verbose = 1, validation_data = (X_test, y_test))
#    print('Training time: %s' %(t-time.time()))
#    (loss, accuracy) = custom_vgg_model.evaluate(X_test, y_test, batch_size = 10, verbose = 1)
#    
#    print('loss=(:.4f), accuracy: {:.4f}%'.format(loss,accuracy * 100))
#    
#    #Saving model in JSON format
#    os.chdir("C://Users//Dhrubo//Desktop//Project//Model")
#    model_json = custom_vgg_model.to_json()
#    with open("final_model.json", "w") as json_file:
#        json_file.write(model_json)
#    # serialize weights to HDF5
#    custom_vgg_model.save_weights("final_model.h5")
#    print("Saved model to disk")
    #load saved model
    json_file = open("C:\\Users\\Noufanpmc\\Documents\\Aegis\\Capstone\\Project\\code\\model_27_96.json",'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("C:\\Users\\Noufanpmc\\Documents\\Aegis\\Capstone\\Project\\code\\model_27_96.h5")
    
    model=loaded_model
    
    print("Loaded Model from disk")
    return render_template("index.html")
    
#=============================================================================#
#global vars for easy reusability
#global model
##
##    
###==================================LOAD MODEL=================================#
#def init(): 
#	json_file = open("E:\VGGFACE JSON\model_27_96.json",'r')
#	loaded_model_json = json_file.read()
#	json_file.close()
#	loaded_model = model_from_json(loaded_model_json)
#	#load woeights into new model
#	loaded_model.load_weights("E:\VGGFACE JSON\model_27_96.h5")
#	print("Loaded Model from disk")
#
##	compile and evaluate loaded model
##	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])	
#	return loaded_model
###=============================================================================#
#model = init()


#================================PREDICTION===================================#
@app.route('/predict/',methods=['POST','GET']) # trigger the URL
def predict():
    return render_template("predict.html")
#def init(): 
#	json_file = open("final_model.json",'r')
#	loaded_model_json = json_file.read()
#	json_file.close()
#	loaded_model = model_from_json(loaded_model_json)
#	#load woeights into new model
#	loaded_model.load_weights("final_model.h5")
#	print("Loaded Model from disk")
#
##	compile and evaluate loaded model
##	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])	
#	return loaded_model 
#
#global model
#model = init()


@app.route('/initiate/',methods=['POST','GET']) # trigger the URL
def initiate():
    
    names=['Aakash','Ahad','Anirudha','Ankita','Archana','Ashwin','Dhruv','Ishani','Manish',
       'Meera','Neha','Noufan','Pooja','Praveen','Raj','Rohit','Shital','Shraddha','Siddharth',
       'Smitha','Sneha','Suhas','Suresh','Taranjit','Tushar','Umesh','Vaibhav']
    video=cv2.VideoCapture(0)
    face_cascade=cv2.CascadeClassifier("C:\\Users\\Noufanpmc\\Documents\\Aegis\\Capstone\\Project\\code\\haarcascade_frontalface_default.xml")
    id=0
    font=cv2.FONT_HERSHEY_SIMPLEX
    a=0
    while True:
        a=a+1
        check, frame=video.read()
        faces=face_cascade.detectMultiScale(frame,scaleFactor=1.30,minNeighbors=10,minSize=(50,50))
        print(faces)                         
        print(check)
        print(frame)
        for x,y,w,h in faces:
            img=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0), 2)
            img1=frame[y:y+h,x:x+w]
            resized_img=cv2.resize(img1,(224,224))
            image = img_to_array(resized_img)
            image = image.reshape((1,image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)
            prediction = model.predict(image)
            
            i=0
            n=27
          for i in range(0,n):
                if (prediction[0][i]>=0.9900):
                    identity=str(i)
                    break;
                else :
                    identity = str(9999)    
                i=i+1
            
            if int(identity) in range(0,n):
                return render_template("success.html")
            else:
                return render_template("failed.html")

                
            #cv2.putText(img,identity,(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),1)#last part=font,red,thickness of text
        #cv2.imshow("Capturing image/video",frame)
        #if (cv2.waitKey(1)==ord('q')):
        break;
    webbrowser.open_new_tab("https://www.muniversity.mobi")

    video.release()
    cv2.destroyAllWindows()

    return render_template("index.html")
#=============================================================================#

    	
#================================INITIALISE===================================#
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='127.0.0.1', port=port)
#=============================================================================#q