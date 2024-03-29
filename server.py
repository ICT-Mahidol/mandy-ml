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
#from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session
from skimage.io import imread
from skimage.transform import resize

import time
import os
from flask import (
    Flask, jsonify, make_response, request, session, render_template
)
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

from flask import Flask, jsonify, request, json
from flask_mysqldb import MySQL
from datetime import datetime
from flask_cors import CORS,cross_origin
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
from flask_jwt_extended import (create_access_token, create_refresh_token, jwt_required, jwt_refresh_token_required, get_jwt_identity, get_raw_jwt)

from flask import abort
from os import environ

from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model



app = Flask(__name__)


app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'mandy'
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
    data_path ='dataset'
    df = pd.read_csv(os.path.join(data_path, 'data.csv'))
    config = ConfigProto()
    config.gpu_options.allow_growth = True  
    sess = Session(config=config)
    json_file = open(r'dataset/model/April-1587174861.5551705.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights(r'dataset/model/April-1587174861.5551705.h5')
    # print(filepath)
    # for i in range(len(filepath)):
    prediction = []
    grad_pred = []
    imagecur = imread(filepath)
    if len(imagecur.shape) == 2:
        imagecur = np.tile(imagecur[:, :, np.newaxis],3)
    
    preprocess = preprocess_input(resize(imagecur, (512, 512)) * 255)
    pred=model.predict(preprocess[np.newaxis, :, :])
    #top_3 = np.argsort(pred[0])[:-4:-1]
    #classes = np.array(df.columns[1:])

    predictions = 'Fracture' if pred[0][0]>0.5 else 'No Fracture'

    
   # prediction = "{}".format(classes[top_3[0]])+"({:.3})".format(pred[0][top_3[0]]*100)
    prediction = "{}".format(predictions)+"({:.3})".format(pred[0][0]*100)
    #print(pred)
    
    for i in range(len(pred)):
        
        #grad_pred.append(classes[pred[i]])
        grad_pred.append(predictions)
        grad_pred = predictions
    
    #top_9 = np.argsort(pred[0])[:-10:-1]
    
    name = []
    prob = []
    for i in range(len(pred)):
        # name.append(classes[pred[i]])
         #prob.append(str("{:.3}".format(top_9[i])))
         
         name.append(predictions)
        
         prob.append(str("{:.3}".format(pred[0][0])))
        

    return prediction,name,prob,grad_pred


def localization(filepath, outpath1, ext, grad_pred):
    IMAGE_PATH = filepath
    LAYER_NAME= 'block5_conv3'
    data = []
    #grad ={'Lt. Condyle':0,'Rt. Condyle':1,'Lt. Ramus-Angle':2,'Rt. Ramus-Angle':3,'Lt. Body':4,'Rt. Body':5,'Lt. Coronoid':6,'Rt. Coronoid':7,'Symphysis-Parasymphysis':8}
    
    #for i in range(len(grad_pred)):
      #for gradName,num in grad.items():
        #if grad_pred[i] == gradName:
           # name = grad_pred[i]
            #CLASS_INDEX = num
    #print(grad_pred)
    name = grad_pred
    CLASS_INDEX = 0          
    json_file = open(r'dataset/model/02042020.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights(r'dataset/model/02042020.h5')

    img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(512, 512))
    img = tf.keras.preprocessing.image.img_to_array(img)

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer('out_relu').output, model.output])

    with tf.GradientTape() as tape:
       conv_outputs, predictions = grad_model(np.array([img]))
       loss = predictions[:, CLASS_INDEX]


    output = conv_outputs[0]
    grads =  tape.gradient(loss, conv_outputs)[0]

    guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

    weights = tf.reduce_mean(guided_grads, axis=(0, 1))

    cam = np.ones(output.shape[0: 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam.numpy(), (512, 512))
    cam = np.maximum(cam, 0)
    r = cam.max() - cam.min()
    if r == 0:
        r = 1;
    heatmap = (cam - cam.min()) / r
    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

    output_image = cv2.addWeighted(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.8, cam, 0.5, 0.5)
    output = cv2.resize(output_image, (1400,800))
    im_rgb = cv2.cvtColor(output , cv2.COLOR_RGB2BGR)
        
    cv2.imwrite(f'{outpath1}_{name}{ext}', im_rgb)
    data.append(f'_{name}{ext}')
    return data
        

    

#@app.route('/users/upload_patient',methods=['POST'])
#def upload_patient():
 #   Patient = request.get_json()['Name']

  #  cur = mysql.connection.cursor()

   # if 'email' in session:
    #        email = session.get('email')

    #cur = mysql.connection.cursor()
	
	
#    cur.execute("INSERT INTO Image (Patient) VALUES ('" + str(Patient) + "')")
   
 #   mysql.connection.commit()



@app.route('/users/upload_annotate',methods=['POST'])
def upload_annotate():
    
    Image_ID = request.get_json()['ID']
    fracture = request.get_json()['Name']
    filepath = request.get_json()['File']

    if 'email' in session:
        email = session.get('email')

    cur = mysql.connection.cursor()

    query = cur.execute("SELECT User_ID from user where Email = email")
	
    cur.execute("INSERT INTO Annotation (User_ID, Image_ID, filepath, fracture) VALUES ('" + 
		str(query) + "', '" + 
		str(Image_ID) + "', '" + 
		str(filepath) + "', '" + 
		str(fracture) + "')")
   
    mysql.connection.commit()

    return 'Done'
    

@app.route('/users/upload_cases', methods=['POST'])
def upload_cases():
    
    if 'file' not in request.files:
        abort(make_response(jsonify(message='There is no file.'), 406));
    files = request.files.getlist('file')
    if len(files) == 0:
        print("err");
        abort(make_response(jsonify(message='There is no file.'), 406));
    
    #results = []
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
        #results.append({
         #   'filename': filename, 
          #  'result': f'Uploaded to {name}-{time.time()}.'})
        #results = 'view result'

        #session['file'] = str(filename)
    
    filename = filename
    timestamp = datetime.utcnow()

    filepath = os.path.join(path, f'source{ext}')

    gradpath1 = os.path.join(path, f'heatmap')

    gradpath_tmp = path+f'/heatmap'
 
    filepath_db = path+f'/source{ext}'
    
   
	
    res_prediction, res_name, res_prob,grad_pred = prediction(filepath)
    gradpath_db = localization(filepath, gradpath1, ext, grad_pred)

    n = ','.join(res_name)
    p = ','.join(res_prob)


    cur = mysql.connection.cursor()
   
    if 'email' in session:
        email = session.get('email')

    query = cur.execute("SELECT User_ID from user where Email = email")
    
    
    cur.execute("INSERT INTO Image (User_ID,filename, timestamp, filepath, prediction) VALUES ('" + 
        str(query) +  "', '" + 
		    str(filename) + "', '" + 
		    str(timestamp) + "', '" + 
		    str(filepath_db ) + "', '" +
           str(res_prediction)+"')")
   
   
    image_ID = cur.execute("SELECT Image_ID FROM image where filename = filename")
    
    
    #for i in range(len(grad_pred)):
    keep = grad_pred
    keep2 = gradpath_tmp+gradpath_db[0]
    print(keep2)
    cur.execute("INSERT INTO gradcam (Image_ID,filepath_heatmap, prediction) VALUES ('" + 
		    str(image_ID) +  "', '" + 
		    str(keep2) + "', '" + 
            str(keep)+"')")


    cur.execute("INSERT INTO percent (Image_ID,prediction, percentage) VALUES ('" + 
		str(image_ID) +  "', '" + 
        str(n) +  "', '" + 
        str(p)+"')")

    mysql.connection.commit()

    results = {'Response':'Result', 'Progress': 100}
    # start running ML?
    return make_response(jsonify(results))


@app.route('/users/get_prediction/<caseId>', methods=['GET'])
def get_predictioin(caseId):
    data = []
    cur = mysql.connection.cursor()

    cur.execute("SELECT prediction FROM percent WHERE Image_ID = (%s)",[caseId])
    for row in cur:
       rv = cur.fetchone()
       prediction = rv['prediction'].split(",")
       for i in range(len(prediction)):
            data.append({'value': prediction[i],'viewValue':prediction[i]})
    return make_response(jsonify(data), 200)

@app.route('/users/get_percent/<caseId>', methods=['GET'])
def get_percent(caseId):
    data = []
    cur = mysql.connection.cursor()
    cur.execute("SELECT prediction, percentage FROM percent WHERE Image_ID = (%s) ",[caseId])
    for row in cur:
        rv = cur.fetchone()
        prediction = rv['prediction'].split(",")
        percentage = rv['percentage'].split(",")
        for i in range(len(prediction)):
            data.append({'name': prediction[i] ,'value': float(percentage[i])*100})
    return make_response(jsonify(data), 200) 

@app.route('/users/get_cases', methods=['GET'])
def get_cases():
    data = []
    pred = []
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM Image")
    for row in cur:
        rv = cur.fetchone()
        data.append({'caseID': rv['Image_ID'], 'caseName': rv['filename'] ,'imageSrc':rv['filepath'],'predDiags': rv['prediction'],'annodiags':'','status':'ready'})
    
    return make_response(jsonify(data), 200)

@app.route('/users/get_image/<caseId>', methods=['GET'])
def get_image(caseId):
    data = []
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM Image WHERE Image_ID = %(caseId)s",{"caseId":[caseId]})
    rv = cur.fetchone()
    data.append({'caseName': rv['filename'],'img': rv['filepath']})
    return make_response(jsonify(data), 200)

@app.route('/users/get_gradcam/<caseId>', methods=['GET'])
def get_gradcam(caseId):
    data = []
    arg1 = request.args['name'];
    
    cur = mysql.connection.cursor()
  
    cur.execute("SELECT A.filename, B.filepath_heatmap, A.Patient FROM image as A  INNER JOIN gradcam as B ON A.Image_ID = B.Image_ID  WHERE B.Image_ID = %(caseId)s AND B.prediction = %(arg1)s",{"caseId":[caseId],"arg1":arg1})
     
    rv = cur.fetchone()
    data.append({'caseName':rv['filename'],'imageSrc':rv['filepath_heatmap']})
   
    # get data from database
    return make_response(jsonify(data), 200)


@app.route('/users/get_annotate/<caseId>', methods=['GET'])
@cross_origin()
def get_annotate(caseId):
    data = []
    cur = mysql.connection.cursor()
    cur.execute("SELECT filename, filepath FROM image where Image_ID = (%s)",[caseId])
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
    #app.run()
    app.secret_key = 'MANDY'