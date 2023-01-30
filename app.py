import os
import shutil
import time

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASEDIR = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp'])

app.config["SECRET_KEY"] = 'adfklasdfkK67986&769row7r1902asdf387132j'
app.config['UPLOAD_PATH'] = os.path.join(BASEDIR, 'static/images')

time.sleep(2.0)
print("start working")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Handle POST Request here
        return render_template('image.html')
    return render_template('image.html')


@app.route('/detect', methods=['GET'])
def detect():
    if request.method == 'GET':
        imgname = request.args.get('imgname')

        shutil.rmtree(BASEDIR + '/UALD/universal_landmark_detection/.eval')
        shutil.rmtree(BASEDIR + '/UALD/runs/GU2Net_runs/results/single_epoch000')

        new_path = BASEDIR + '/UALD/data/08_11/pngs/' + imgname
        old_path = BASEDIR + '/static/images/' + imgname
        shutil.copy(old_path, new_path)

        yolo_path = 'main.py'

        os.system('cd UALD/universal_landmark_detection && python ' + yolo_path + ' -f ' + imgname)

        before_json = imgname.rfind('.')
        file_name = imgname[:before_json]
        img_with_dots = file_name + '.png'

        txt_path = BASEDIR + '/UALD/universal_landmark_detection/.eval/.._runs_GU2Net_runs_results_single_epoch000/chest/images/' + file_name + '.txt'
        with open(txt_path, 'r') as f:
            angles = f.readlines()

        result_old_path = BASEDIR + '/UALD/universal_landmark_detection/.eval/.._runs_GU2Net_runs_results_single_epoch000/chest/images/'
        result_new_path = BASEDIR + '/static/images/'
        shutil.copy(result_old_path + img_with_dots, result_new_path + img_with_dots)

        angle_name = ['CE角', '臼顶倾斜角', 'Sharp角', '头臼指数']
        reference = ['20°-40°', '<10°', '≤40°', '>75']
        left = angles[:4]
        right = angles[4:]
        for i in range(len(left) - 1):
            left[i] = left[i] + '°'
        for i in range(len(right) - 1):
            right[i] = right[i] + '°'
        angle_value = zip(angle_name, reference, left, right)

        return render_template('image.html', imgname=imgname, img_with_dots=img_with_dots, angle_value=angle_value)


@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        filepath = ''
        f = request.files['image']
        if f and allowed_file(f.filename):
            imgname = secure_filename(f.filename)
            filepath = os.path.join(app.config['UPLOAD_PATH'], imgname)
            print('filepath:{}'.format(filepath))
            f.save(filepath)

            return render_template("image.html", imgname=imgname)
        return render_template("image.html")


if __name__ == '__main__':
    # DEBUG is SET to TRUE. CHANGE FOR PROD
    app.run(port=5000, debug=True)
