#Library Imports
from flask import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model, Sequential
import numpy as np
import pandas as pd
import shutil
import time
import cv2 as cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import os
import seaborn as sns
sns.set_style('darkgrid')
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from IPython.display import display, HTML
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import werkzeug


def classify(sdir, csv_path,  model_path, crop_image = False):    
    # read in the csv file
    class_df=pd.read_csv(csv_path)    
    img_height=int(class_df['height'].iloc[0])
    img_width =int(class_df['width'].iloc[0])
    img_size=(img_height, img_width)
    scale=class_df['scale by'].iloc[0] 
    try: 
        s=int(scale)
        s2=1
        s1=0
    except:
        split=scale.split('-')
        s1=float(split[1])
        s2=float(split[0].split('*')[1]) 
        print (s1,s2)
    path_list=[]
    paths=os.listdir(sdir)
    for f in paths:
        path_list.append(os.path.join(sdir,f))
    print (' Model is being loaded- this will take about 10 seconds')
    model=load_model(model_path)
    image_count=len(path_list)    
    index_list=[] 
    prob_list=[]
    cropped_image_list=[]
    good_image_count=0
    for i in range (image_count):       
        img=plt.imread(path_list[i])
        if crop_image == True:
            status, img=crop(img)
        else:
            status=True
        if status== True:
            good_image_count +=1
            img=cv2.resize(img, img_size)            
            cropped_image_list.append(img)
            img=img*s2 - s1
            img=np.expand_dims(img, axis=0)
            p= np.squeeze (model.predict(img))           
            index=np.argmax(p)            
            prob=p[index]
            index_list.append(index)
            prob_list.append(prob)
    if good_image_count==1:
        class_name= class_df['class'].iloc[index_list[0]]
        probability= prob_list[0]
        img=cropped_image_list [0] 
        plt.title(class_name, color='blue', fontsize=16)
        plt.axis('off')
        plt.imshow(img)
        return class_name, probability
    elif good_image_count == 0:
        return None, None
    most=0
    for i in range (len(index_list)-1):
        key= index_list[i]
        keycount=0
        for j in range (i+1, len(index_list)):
            nkey= index_list[j]            
            if nkey == key:
                keycount +=1                
        if keycount> most:
            most=keycount
            isave=i             
    best_index=index_list[isave]    
    psum=0
    bestsum=0
    for i in range (len(index_list)):
        psum += prob_list[i]
        if index_list[i]==best_index:
            bestsum += prob_list[i]  
    img= cropped_image_list[isave]/255    
    class_name=class_df['class'].iloc[best_index]
    plt.title(class_name, color='blue', fontsize=16)
    plt.axis('off')
    plt.imshow(img)
    return class_name, bestsum/image_count


#Our Flask App
app = Flask(__name__)

@app.route('/scan', methods = ['POST','GET'])
def scan():
    if request.method == 'POST':
        #Recieving files from the device
        imagefile = request.files['image']
        custom_name = 'check.jpg'  # replace this with your desired name and file extension
        imagefile.save("C:/Users/u2021605/Desktop/Classifier Server/" + custom_name)


        #Settling paths of image, model and csv
        working_dir  = 'C:/Users/u2021605/Desktop/Classifier Server/'
        store_path=os.path.join(working_dir, 'storage')
        if os.path.isdir(store_path):
            shutil.rmtree(store_path)
        os.mkdir(store_path)
        img_path='C:/Users/u2021605/Desktop/Classifier Server/check.jpg'
        img=cv2.imread(img_path)
        file_name=os.path.split(img_path)[1]
        dst_path=os.path.join(store_path, file_name)
        cv2.imwrite(dst_path, img)
        print (os.listdir(working_dir))
        mango_dict = {
        'Anwar Ratool': 300,
        'Chaunsa (Black)': 314,
        'Chaunsa (Summer Bahisht)': 230,
        'Chaunsa (White)': 400,
        'Dosehri': 123,
        'Fajri': 435,
        'Langra': 223,
        'Sindhri': 342}


        #Our Trained Model Being Loaded and Applied
        csv_path='' # path to class_dict.csv
        model_path='' # path to the trained model
        class_name, probability=classify(store_path, csv_path,  model_path, crop_image = False) # run the classifier
        cost = mango_dict.get(class_name)
        msg=f'I am {probability * 100: 6.2f} % sure that this is {class_name} and today its cost is: {cost}/-'

        print(msg)
        return jsonify({"message":msg})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port='80')