import os, random, time
from werkzeug import secure_filename
from flask import Flask, request, url_for, send_from_directory, g, jsonify
from flask import flash, render_template
from flask import make_response, redirect

import PIL
from PIL import Image

import joblib
from keras.models import Model
from keras.models import load_model

import os
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import numpy as np
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras import optimizers
import tensorflow as tf
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.regularizers import l1, l2
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import to_categorical
from keras import initializers
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import argparse

from imageio import imread
from skimage.transform import resize

#import waterquality_Flask.exif_gps
import waterquality_Flask.model_predict
import waterquality_Flask.predict

ALLOWED_EXTENSIONS = set(['png','PNG', 'jpg', 'JPG','jpeg','JPEG', 'gif', 'pdf'])

app = Flask(__name__)

app.config['SECRET_KEY'] = 'I have a dream'
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd()
app.config['UPLOAD_FOLDER'] = os.getcwd()

# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

#photos = UploadSet('photos', IMAGES)
#configure_uploads(app, photos)
#patch_request_class(app)  # set maximum file size, default is 16MB

thumbnail_dir = os.path.join(os.getcwd(), "waterquality_Flask", "static", "img", "thumbnail")
model_link = 'waterquality_Flask/supporting_files/weights_mobileNet_without_pre1570856691.343963_color.h5'
scaler_link = 'waterquality_Flask/supporting_files/MaxAbsScaler.pkl'

# class UploadForm(FlaskForm):
#     photo = FileField(validators=[
#         FileAllowed(photos, u'format not supported'),
#         FileRequired(u'havent choose photo yet')])
#     submit = SubmitField(u'upload')


# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     form = UploadForm()
#     if form.validate_on_submit():
#         filename = photos.save(form.photo.data)
#         file_url = photos.url(filename)
#     else:
#         file_url = None
#     return render_template('index.html', form=form, file_url=file_url)


@app.before_request
def before_request():
    g.file_url = None


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'],
#                                filename)

# @app.route('/retrive_geolocation')
# def retrive_geolocation():
#     imgurl = request.cookies.get('imgurl')
#     resp = make_response(redirect(url_for("endpoint", imgurl='')))
#     lat, lon = exif_gps.get_exif_location(imgurl)

#     resp.set_cookie('imGeolocation', 'lat:'+str(lat)+', '+'lon'+str(lon))
#     print(lat, lon)
#     return resp


@app.route('/resetimage', methods=['GET', 'POST'])
def resetimage():
    resp = make_response(redirect(url_for("endpoint", imgurl='')))
    resp.set_cookie('imgurl', '')
    resp.set_cookie('prediction', '')
    # resp.set_cookie('username', 'the username')
    return resp


@app.route("/")
def endpoint(imgurl='', prediction=-1):
    # form = UploadForm()
    imgurl = request.cookies.get('imgurl')
    prediction = request.cookies.get('prediction')
    imGeolocation = request.cookies.get('imGeolocation')
    # return make_response(render_template("index.html", form=form, imgurl=imgurl, prediction=prediction,imGeolocation=imGeolocation, items=[]))
    return make_response(render_template("index.html", items=[]))


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    items = []
    images = request.files.getlist('images') #convert multidict to dict
    img_dir_path = os.path.join(os.getcwd(), "waterquality_Flask", "web_img")

    scaler = joblib.load(scaler_link)
    model = load_model(model_link)

    if images is not None:
        for image in images:
            file_name = secure_filename(image.filename)
            print("[image] {}".format(file_name))
            
            current_path = os.path.join(img_dir_path, file_name)
            image.save(current_path)
            print("[image path] {}".format(current_path))
           
           # imgurl = url_for('uploaded_file', filename=file_name)
            # print(imgurl)
            # prediction = train_and_predict(data_path=file_name)
            # Keras==2.3.0
            prediction = model_predict.predict(model, scaler, current_path)
            prediction = str(prediction)
            
            #lat, lon = exif_gps.get_exif_location(file_name)
            
            #imtime = exif_gps.get_datetime(file_name)
            
            #imGeolocation = 'lat:'+str(lat)+', '+'lon'+str(lon)
            imGeolocation = ""
            imtime = ""

            thumbnail_path = create_thumbnail(current_path, thumbnail_dir)
            #imgurl_t = url_for('uploaded_file', filename=imgurl_t_file_name)
            items.append((prediction, thumbnail_path, imGeolocation, imtime))
            os.remove(current_path)
            # resp.set_cookie('imGeolocation', 'lat:'+str(lat)+', '+'lon'+str(lon))
            # resp.set_cookie('imgurl', imgurl)
            # resp.set_cookie('imtime', imtime)
            # resp.set_cookie('prediction', prediction)
    resp = make_response(render_template("index.html", items=items))
    # form = UploadForm()
    # if form.validate_on_submit():
    #     filename = photos.save(form.photo.data)
    #     file_url = photos.url(filename)
    #     lat, lon = exif_gps.get_exif_location(file_url.split("/")[-1])
    #     print(file_url)
    #     print(lat, lon)
    # else:
    #     file_url = ''
    #     lat, lon = '', ''
    # # app.logger.info(file_url)
    # print(file_url)
    # prediction = request.cookies.get('prediction')
    # resp = make_response(render_template("index.html",  form=form, imgurl=file_url, file_url=file_url, prediction=prediction, imGeolocation='lat:'+str(lat)+', '+'lon'+str(lon)))
    # resp.set_cookie('imgurl', file_url)
    # resp.set_cookie('imGeolocation', 'lat: '+str(lat)+', '+'lon: '+str(lon))
    return resp

@app.route('/webAPI', methods=['POST'])
def web_upload():
    images = request.files.getlist('images') #convert multidict to dict
    items = []
    error_desc = ""

    scaler = joblib.load(scaler_link)
    model = load_model(model_link)

    if images:
        img_dir_path = os.path.join(os.getcwd(), "waterquality_Flask", "web_img")
        for image in images:
            # get fileName
            file_name = secure_filename(image.filename)
            print("[image] {}".format(file_name))

            # get filePath
            current_path = os.path.join(img_dir_path, file_name)
            image.save(current_path)
            print("[image path] {}".format(current_path))

            # predict the given image path (Keras==2.3.0)
            prediction = model_predict.predict(model, scaler, current_path)
            prediction = str(prediction)
            print("[image predict] {}".format(prediction))

            # handle location
            imGeolocation = ""

            # handle time
            imtime = ""

            # handle thumbnail
            thumbnail_path = create_thumbnail(current_path, thumbnail_dir)

            # marshal together
            item = {"prediction": prediction, "time": imtime, "location": imGeolocation, "thumbnail": thumbnail_path}

            items.append(item)
            
            os.remove(current_path)
            # lat, lon = exif_gps.get_exif_location(file_name)
            # imtime = exif_gps.get_datetime(file_name)
            # imGeolocation = 'lat:'+str(lat)+', '+'lon'+str(lon)

            # imgurl_t_file_name = create_thumbnail(file_name)
            # imgurl_t = url_for('uploaded_file', filename=imgurl_t_file_name)
            # items.append((prediction, imgurl_t, imGeolocation, imtime))

            # delete image after prediction work
            # resp.set_cookie('imGeolocation', 'lat:'+str(lat)+', '+'lon'+str(lon))
            # resp.set_cookie('imgurl', imgurl)
            # resp.set_cookie('imtime', imtime)
            # resp.set_cookie('prediction', prediction)
        return jsonify(results = items, error = False, error_description = error_desc)
    else:
        error_desc = "No images"
        return jsonify(results = items, error = True, error_description = error_desc)



@app.route('/app', methods=['POST'])
def mobile_upload():
    images = request.files.getlist('images') #convert multidict to dict
    img_dir_path = os.path.join(os.getcwd(), "waterquality_Flask", "app_img")

    items = []
    error_desc = ""

    def load_smoother(path):
        u_model = joblib.load(path+'/upper_model.pkl')
        l_model = joblib.load(path+'/lower_model.pkl')
        return u_model,l_model
    
    u_model,l_model = load_smoother(r'waterquality_Flask/supporting_files')

    scaler = joblib.load(scaler_link)
    model = load_model(model_link)
    
    if images:
        for image in images:

            file_name = secure_filename(image.filename)
            print("[image] {}".format(file_name))
            current_path = os.path.join(img_dir_path, file_name)
            image.save(current_path)
            print("[image path] {}".format(current_path))

            # Keras==2.3.0

            prediction = model_predict.predict(model, scaler, current_path)

            upper_offset = u_model.predict(np.array(prediction, np.float).reshape((-1, 1)))[0]
            lower_offset = l_model.predict(np.array(prediction, np.float).reshape((-1, 1)))[0]

            print("[image predict] {}".format(prediction))
            print("[image predict lower range] {}".format(lower_offset))
            print("[image predict upper range] {}".format(upper_offset))
            print('predict range is [{:f}, {:f}]'.format(prediction - lower_offset, prediction + upper_offset))

            lat, lon = round(random.uniform(-90, 90), 2), round(random.uniform(-90, 90), 2)
            position = str(lat) + ", " + str(lon)

            thumbnail_path = "static/img/thumbnail/0061a_t.JPG"
            
            item = {"prediction": str(prediction), "lower_range": str(lower_offset), "upper_range": str(upper_offset), "time": "", "location": position, "thumbnail": thumbnail_path}

            items.append(item)

            # delete image after prediction work
            os.remove(current_path)
        return jsonify(results = items, error = False, error_description = error_desc)
    else:
        error_desc = "No images"
        return jsonify(results = items, error = True, error_description = error_desc)


@app.route('/testAPI', methods=['POST'])
def test_upload():
    images = request.files.getlist('images') #convert multidict to dict
    items = []
    error_desc = ""
    img_dir_path = os.path.join(os.getcwd(), "waterquality_Flask", "app_img")
    if images:
        for image in images:

            file_name = secure_filename(image.filename)
            print("[image] {}".format(file_name))
            current_path = os.path.join(img_dir_path, file_name)
            image.save(current_path)
            print("[image path] {}".format(current_path))

            time.sleep(random.randint(4, 10))
            prediction = round(random.uniform(1, 500), 2)
            prediction = str(prediction)
            print("[image test predicton] {}".format(prediction))

            lat, lon = round(random.uniform(-90, 90), 2), round(random.uniform(-90, 90), 2)
            position = str(lat) + ", " + str(lon)

            thumbnail_path = "static/img/thumbnail/0061a_t.JPG"
            
            item = {"prediction": prediction, "time": "", "location": position, "thumbnail": thumbnail_path}

            items.append(item)

            # delete image after prediction work
            os.remove(current_path)

        return jsonify(results = items, error = False, error_description = error_desc)
    else:
        error_desc = "No images"
        return jsonify(results = items, error = True, error_description = error_desc)

# @app.route('/calculating', methods=['GET', 'POST'])
# def calculating():
#     print("in calculating")
#     print(g.file_url)
#     imgurl = request.cookies.get('imgurl')
#     prediction = str(train_and_predict(data_path=imgurl))
#     # resp = make_response(render_template("index.html", prediction=1))
#     response = make_response(redirect(url_for("endpoint", prediction=1, imgurl=imgurl)))

#     response.set_cookie('prediction', prediction)
#     print(imgurl)
#     return response

def create_thumbnail(image_path, dir_path):
    filename, ext = os.path.splitext(os.path.basename(image_path))
    base_width = 300
    img = Image.open(image_path)
    pre_width, pre_height = img.size
    if pre_width > pre_height:
        pre_width, pre_height = pre_height, pre_width
        
    if pre_width <= 300:
        file_path = os.path.join(dir_path, filename + '_t' + ext)
        img.save(file_path)
        return file_path

    print("[thumbnail] pre_width: {}, pre_height: {}".format(pre_width, pre_height))

    base_height = int(base_width / float(pre_width) * float(pre_height))

    print("[thumbnail] cal_base_width: {}, cal_height: {}".format(base_width, base_height))

    img = img.resize((base_width, base_height), PIL.Image.ANTIALIAS)

    print("[thumbnail] after_width: {}, after_height: {}".format(img.size[0], img.size[1]))

    file_path = os.path.join(dir_path, filename + '_t' + ext)
    img.save(file_path)
    # hard code, can't find a way to parse to the correct static/img/thumbnail folder
    return "static/img/thumbnail/" + filename + '_t' + ext

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888,threaded=True)



