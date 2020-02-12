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
app.config['IMAGE_HEAT'] = os.path.join('static/','grad/')
app.config['IMAGE_EXT'] = {'.png', '.jpg', '.jpeg'}


app.secret_key = 'MANDY'


mysql = MySQL(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

CORS(app, supports_credentials=True)



def prediction(filepath):
    data_path ='dataset'
    df = pd.read_csv(os.path.join(data_path, 'data.csv'))
    config = ConfigProto()
    config.gpu_options.allow_growth = True  
    sess = Session(config=config)
    json_file = open(r'dataset/model/classifier-2601-1580033241.0532682.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights(r'dataset/model/classifier-2601-1580033241.0532682.h5')
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
    
    top_8 = np.argsort(pred[0])[:-9:-1]

   
    #image_ID = cur.execute("SELECT Image_ID FROM image where filepath = filepath")
    
    name = []
    prob = []
    for i in range(8):
        
        name.append(classes[top_8[i]])
        prob.append(pred[0][top_8[i]])

#image_ID = cur.execute("SELECT Image_ID FROM image where filepath = filepath")
        #cur.execute("INSERT INTO percent (Image_ID,prediction, percentage) VALUES ('" + 
		 #   str(image_ID) +  "', '" + 
          #  str(name) +  "', '" + 
           # str(prob)+"')")
   
        mysql.connection.commit()
        

    return prediction,name,prob


@app.route('/users/get_percent/<caseId>', methods=['GET'])
def get_percent(caseId):
    data = []
    cur = mysql.connection.cursor()
    cur.execute("SELECT prediction, percentage FROM percent WHERE Image_ID = (%s) ",caseId)
    for row in range(8):
        rv = cur.fetchone()
        data.append({'name': rv['prediction'] ,'value': float(rv['percentage'])*100})
    return make_response(jsonify(data), 200)


def localization(filepath, outpath):
    IMAGE_PATH = filepath
    LAYER_NAME= 'block_16_project'
    CLASS_INDEX = 9

    img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)

    model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=True)

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(LAYER_NAME).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([img]))
        
        loss = predictions[:, CLASS_INDEX]
 
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

    weights = tf.reduce_mean(guided_grads, axis=(0, 1))

    cam = np.ones(output.shape[0: 2], dtype = np.float32)

    @tf.function
    def loop(weights):
        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

    output_image = cv2.addWeighted(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam, 1, 0)
    output = cv2.resize(output_image, (800,400))

    cv2.imwrite(outpath, output)
    




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

        session['file'] = str(filename)
    
    filename = filename
    filepath = os.path.join(path, f'source{ext}')
    gradpath = os.path.join(path, f'heatmap{ext}')

    filepath_db = path+f'/source{ext}'
    gradpath_db = path+f'/heatmap{ext}'
    timestamp = datetime.utcnow()
	
    
    res_prediction, res_name, res_prob = prediction(filepath)

    localization(filepath, gradpath)
   

    temp = ""
    for i in range(len(res_prediction)):
        if i==len(res_prediction):
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
   
   
    image_ID = cur.execute("SELECT Image_ID FROM image where filename = filename")
    
    for i in range(len(res_prediction)):
        keep = res_prediction[i]
        cur.execute("INSERT INTO gradcam (Image_ID,filepath_heatmap, prediction) VALUES ('" + 
		    str(image_ID) +  "', '" + 
		    str(gradpath_db) + "', '" + 
            str(keep)+"')")

    for i in range(8):
            
        name = res_name[i]
        prob = res_prob[i]

        cur.execute("INSERT INTO percent (Image_ID,prediction, percentage) VALUES ('" + 
		    str(image_ID) +  "', '" + 
            str(name) +  "', '" + 
            str(prob)+"')")

    
    mysql.connection.commit()

   
    # start running ML?
    return make_response(jsonify(results), 201)


@app.route('/users/get_prediction/<caseId>', methods=['GET'])
def get_predictioin(caseId):
    data = []
    cur = mysql.connection.cursor()

    cur.execute("SELECT prediction FROM percent WHERE Image_ID = (%s)", caseId)
    i = 0;
    for row in range(3):
        rv = cur.fetchone()
        pred = rv['prediction']+'-'+str(i)
        data.append({'value': pred ,'viewValue':rv['prediction']})
        i= i+1
       
    # get data from database
    #print(data)
    return make_response(jsonify(data), 200)
   

@app.route('/users/get_cases', methods=['GET'])
def get_cases():
    data = []
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM Image")
    for row in cur:
        rv = cur.fetchone()
        data.append({'caseID': rv['Image_ID'], 'caseName': rv['filename'] ,'imageSrc':rv['filepath'],'predDiags': rv['prediction'],'annodiags':'','status':'ready'})
    
    return make_response(jsonify(data), 200)

@app.route('/users/get_gradcam/<caseId>', methods=['GET'])
def get_gradcam(caseId):
    data = []
    cur = mysql.connection.cursor()
    
    cur.execute("SELECT A.filename, B.filepath_heatmap FROM image as A INNER JOIN gradcam as B ON A.Image_ID = B.Image_ID  WHERE B.Image_ID = (%s)",caseId)
    for row in cur:
        rv = cur.fetchone()
        data.append({'caseName':rv['filename'],'imageSrc':rv['filepath_heatmap']})
   
    # get data from database
    return make_response(jsonify(data), 200)


@app.route('/users/get_annotate/<caseId>', methods=['GET'])
def get_annotate(caseId):
    data = []
    cur = mysql.connection.cursor()
    cur.execute("SELECT filename, filepath FROM image where Image_ID = (%s)",caseId)
    #for row in cur:
    rv = cur.fetchone()
    heat = rv['filepath']
    data.append({'caseName':rv['filename'],'imageSrc':rv['filepath']})
        
       
    # get data from database
    return make_response(jsonify(heat), 200)


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
    



