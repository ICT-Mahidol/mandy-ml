import time
import os
from flask import (
    Flask, jsonify, make_response, request
)
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['IMAGE_DIR'] = os.path.join('static', 'cases')
app.config['IMAGE_EXT'] = {'.png', '.jpg', '.jpeg'}
CORS(app, supports_credentials=True)


@app.route('/status')
def status():
    return 'mandy server is running.'


@app.route('/upload_cases', methods=['POST'])
def upload_cases():
    if 'file' not in request.files:
        abort(make_response(jsonify(message='There is no file.'), 406))
    files = request.files.getlist('file')
    if len(files) == 0:
        abort(make_response(jsonify(message='There is no file.'), 406))
    
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

    # start running ML?
    return make_response(jsonify(results), 201)


if __name__ == '__main__':
    app.run()