import os
import werkzeug
from flask import Flask, request, url_for, send_from_directory, g, jsonify
from flask import flash, render_template
from flask import make_response, redirect

#from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
#from flask_wtf import FlaskForm
#from flask_wtf.file import FileField, FileRequired, FileAllowed
# from wtforms import SubmitField

import PIL
from PIL import Image

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
            prediction = model_predict.predict(current_path)
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

@app.route('/app', methods=['POST'])
def mobile_upload():
    images = request.files.getlist('images') #convert multidict to dict
    values = []
    img_dir_path = os.path.join(os.getcwd(), "waterquality_Flask", "app_img")
    if images:
        for image in images:
            file_name = secure_filename(image.filename)
            print("[image] {}".format(file_name))

            current_path = os.path.join(img_dir_path, file_name)

            image.save(current_path)
            print("[image path] {}".format(current_path))

            # Keras==2.3.0
            prediction = model_predict.predict(current_path)
            prediction = str(prediction)
            print("[image predict] {}".format(prediction))

            # lat, lon = exif_gps.get_exif_location(file_name)
            # imtime = exif_gps.get_datetime(file_name)
            # imGeolocation = 'lat:'+str(lat)+', '+'lon'+str(lon)

            # imgurl_t_file_name = create_thumbnail(file_name)
            # imgurl_t = url_for('uploaded_file', filename=imgurl_t_file_name)
            # items.append((prediction, imgurl_t, imGeolocation, imtime))
            values.append(prediction)

            # delete image after prediction work
            os.remove(current_path)
            # resp.set_cookie('imGeolocation', 'lat:'+str(lat)+', '+'lon'+str(lon))
            # resp.set_cookie('imgurl', imgurl)
            # resp.set_cookie('imtime', imtime)
            # resp.set_cookie('prediction', prediction)
    
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
    return jsonify(results = values)


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
    if img.size[0] <= 300:
        file_path = os.path.join(dir_path, filename + '_t' + ext)
        img.save(file_path)
        return file_path
    w_percent = (base_width / float(img.size[0]))
    h_size = int((float(img.size[1]) * float(w_percent)))
    img = img.resize((base_width, h_size), PIL.Image.ANTIALIAS)
    file_path = os.path.join(dir_path, filename + '_t' + ext)
    img.save(file_path)
    # hard code, can't find a way to parse to the correct static/img/thumbnail folder
    return "static/img/thumbnail/" + filename + '_t' + ext

# http://34.73.100.1:8000/waterquality_Flask/static/img/thumbnail/image_92181_t.jpg
# http://34.73.100.1:8000/static/img/parallax/testimonial.jpg

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888,threaded=True)



