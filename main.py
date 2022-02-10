from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
from torch_utils import predict

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/result", methods=['POST'])
def upload():
    file = request.files['file']
    
    if file.filename == "":
        flash("no image selecting for uploading")
        return redirect(url_for("index")) 

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)
            new_filename = predict(img_path)
            return render_template('index.html', filename_1=filename, filename_2 = new_filename)
        except:
            flash('Something went wrong :(')
            return redirect(url_for("index")) 
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(url_for("index")) 

@app.route('/display/<filename>')
def result(filename):
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__=="__main__":
    app.run(debug=True)