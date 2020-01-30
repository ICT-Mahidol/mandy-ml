import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
import redis
import cv2

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import IPython.display as display
from PIL import Image

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session
from skimage.io import imread
from skimage.transform import resize

import time
import os
from flask import (
    Flask, jsonify, make_response, request, session
)
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

from flask import Flask, jsonify, request, json
from flask_mysqldb import MySQL
from datetime import datetime
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
from flask_jwt_extended import (create_access_token, create_refresh_token, jwt_required, jwt_refresh_token_required, get_jwt_identity, get_raw_jwt)

from flask import abort
from os import environ

app = Flask(__name__)

app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'Mandy'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
app.config['JWT_SECRET_KEY'] = 'secret'

app.config['IMAGE_DIR'] = os.path.join('static/', 'cases/')
app.config['IMAGE_EXT'] = {'.png', '.jpg', '.jpeg'}




app.secret_key = 'MANDY'





mysql = MySQL(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

CORS(app, supports_credentials=True)



def prediction(filepath):
    data_path ='C:/Users/Admin/mandy-app/mandy-app-master/mandy-ml-master/dataset'
    df = pd.read_csv(os.path.join(data_path, 'data.csv'))
    config = ConfigProto()
    config.gpu_options.allow_growth = True  
    sess = Session(config=config)
    json_file = open(r'C:/Users/Admin/mandy-app/mandy-app-master/mandy-ml-master/dataset/model/classifier-2601-1580033241.0532682.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights(r'C:/Users/Admin/mandy-app/mandy-app-master/mandy-ml-master/dataset/model/classifier-2601-1580033241.0532682.h5')
    # print(filepath)
    # for i in range(len(filepath)):
    prediction = []
    imagecur = imread(filepath)
    adjustimg = cv2.GaussianBlur(imagecur,(5,5),0)
    if len(adjustimg.shape) == 2:
        adjustimg = np.tile(adjustimg[:, :, np.newaxis], 3)
    preprocess = preprocess_input(resize(adjustimg, (512, 512)) * 255)
    pred=model.predict(preprocess[np.newaxis, :, :])
    top_3 = np.argsort(pred[0])[:-4:-1]
    classes = np.array(df.columns[1:])
    for i in range(3):
        #print("{}".format(classes[top_3[i]])+" ({:.3})".format(pred[0][top_3[i]]))
        prediction.append("{}".format(classes[top_3[i]])+" ({:.3})".format(pred[0][top_3[i]]))
    #print("\n")
    return prediction
        
            

@app.route('/users/upload_cases', methods=['POST'])
def upload_cases():
    
    if 'file' not in request.files:
        abort(make_response(jsonify(message='There is no file.'), 406));
    files = request.files.getlist('file')
    if len(files) == 0:
        print("err");
        abort(make_response(jsonify(message='There is no file.'), 406));
    
    results = []
    for file in files:
        
        filename = secure_filename(file.filename).strip()
        name, ext = os.path.splitext(filename)
        name = name.strip()
        ext = ext.strip().lower()
        if filename == '' or name == '':
            results.append(
                {'filename': 'Untitled', 'result': 'Empty file name.'})
            continue
        if ext not in app.config['IMAGE_EXT']:
            results.append(
                {'filename': filename, 'result': 'Incorrect file extension.'})
            continue
        path = os.path.join(
            app.config['IMAGE_DIR'], f'{name}-{time.time()}')
        os.mkdir(path)
        file.save(os.path.join(path, f'source{ext}'))
        results.append({
            'filename': filename, 
            'result': f'Uploaded to {name}-{time.time()}.'})
    filename = filename
    filepath = os.path.join(path, f'source{ext}')
    filepath_db = path+f'/source{ext}'
    timestamp = datetime.utcnow()
	
    
    res_prediction = prediction(filepath)
    temp = ""
    for i in range(len(res_prediction)):
        if i==len(res_prediction)-1:
            temp += res_prediction[i]
        temp += res_prediction[i] +','
       
   
    cur = mysql.connection.cursor()
   
    

    if 'email' in session:
        email = session.get('email')
   
    query = cur.execute("SELECT User_ID from user where Email = email")
    
    
    cur.execute("INSERT INTO Image (User_ID,filename, timestamp, filepath, prediction) VALUES ('" + 
		    str(query) +  "', '" + 
		    str(filename) + "', '" + 
		    str(timestamp) + "', '" + 
		    str(filepath_db ) + "', '" +
            str(temp)+"')")
   
    mysql.connection.commit()
   
    # start running ML?
    return make_response(jsonify(results), 201)
   

@app.route('/users/get_cases', methods=['GET'])
def get_cases():

    
    data = []
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM Image")
    for row in cur:
        rv = cur.fetchone()
        data.append({'caseName': rv['filename'] ,'imageSrc':rv['filepath'],'predDiags':rv['prediction'],'annodiags':'','status':'ready'})
        
       
    # get data from database
    print(data)
    return make_response(jsonify(data), 200)


@app.route('/users/register', methods=['POST'])
def register():
    cur = mysql.connection.cursor()
    first_name = request.get_json()['first_name']
    last_name = request.get_json()['last_name']
    email = request.get_json()['email']
    password = bcrypt.generate_password_hash(request.get_json()['password']).decode('utf-8')
    created = datetime.utcnow()
	
    cur.execute("INSERT INTO user (Firstname, Lastname, Email, Password) VALUES ('" + 
		str(first_name) + "', '" + 
		str(last_name) + "', '" + 
		str(email) + "', '" + 
		str(password) + "')")
   
    mysql.connection.commit()
	
    result = {
		'first_name' : first_name,
		'last_name' : last_name,
		'email' : email,
		'password' : password
	}

    return jsonify({'result' : result})
	

@app.route('/users/login', methods=['POST'])
def login():
    cur = mysql.connection.cursor()
    email = request.get_json()['email']
    password = request.get_json()['password']
    result = ""
    
    session['email'] = str(email)
    cur.execute("SELECT * FROM user where Email = '" + str(email) + "'")
    rv = cur.fetchone()
	
    if bcrypt.check_password_hash(rv['Password'], password):
        access_token = create_access_token(identity = {'first_name': rv['Firstname'],'last_name': rv['Lastname'],'email': rv['Email']})
        result = jsonify({"token":access_token})
    else:
        result = jsonify({"error":"Invalid username and password"})
    
    return result
    
@app.route('/users/status')
def status():
    return 'mandy server is running.'


if __name__ == '__main__':
    app.run(debug=True)
    app.secret_key = 'MANDY'
    



