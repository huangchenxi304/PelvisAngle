import sys

from flask import Flask, redirect, url_for, render_template, request, Response
import os
import cv2
import time
import shutil
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename

from typing import Dict, List

app = Flask(__name__)

BASEDIR = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp'])

app.config["SECRET_KEY"] = 'adfklasdfkK67986&769row7r1902asdf387132j'
app.config['UPLOAD_PATH'] = os.path.join(BASEDIR, 'static/images')
# app.config['RESULT_PATH'] = os.path.join(BASEDIR, 'UALD\universal_landmark_detection\.eval\.._runs_GU2Net_runs_results_single_epoch000\chest\images')

# 初始化摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 680)  # 设置摄像头输出宽
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)  # 设置摄像头输出高
print("start reading video...")
time.sleep(2.0)
print("start working")


# emotion_detector = EmotionDetector(
#         model_loc="models",
#         face_detection_threshold=0.8,
#         face_detector="dlib",
#     )

def load_emojis(path: str = "data//emoji") -> List:
    emojis = {}

    # list of given emotions
    EMOTIONS = [
        "Angry",
        "Disgusted",
        "Fearful",
        "Happy",
        "Sad",
        "Surprised",
        "Neutral",
    ]

    # store the emoji coreesponding to different emotions
    for _, emotion in enumerate(EMOTIONS):
        emoji_path = os.path.join(path, emotion.lower() + ".png")

        emojis[emotion] = cv2.imread(emoji_path, -1)

    return emojis


emoji_loc = "data"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Handle POST Request here
        return render_template('image.html')
    return render_template('image.html')


@app.route('/video_file')
def video_file():
    return render_template('video_file.html')


@app.route('/detect', methods=['GET'])
def detect():
    if request.method == 'GET':
        imgname = request.args.get('imgname')
        inputtype = request.args.get('type')
        from emotion_analyzer.media_utils import load_image_path

        # ob = EmotionAnalysisVideo(
        #     face_detector="dlib",
        #     model_loc="models",
        #     face_detection_threshold=0.0,
        # )
        shutil.rmtree(BASEDIR + '/UALD/universal_landmark_detection/.eval')
        shutil.rmtree(BASEDIR + '/UALD/runs/GU2Net_runs/results/single_epoch000')
        # img1 = load_image_path(os.path.join(app.config['UPLOAD_PATH'], imgname))
        # emotion, emotion_conf = ob.emotion_detector.detect_facial_emotion(img1)
        new_path = BASEDIR + '/UALD/data/08_11/pngs/' + imgname
        old_path = BASEDIR + '/static/images/' + imgname
        shutil.copy(old_path, new_path)

        yolo_path = 'main.py'
        print('start')

        os.system('cd UALD/universal_landmark_detection && python ' + yolo_path + ' -f ' + imgname)

        print('end')
        before_json = imgname.rfind('.')
        file_name = imgname[:before_json]
        img_with_dots = file_name + '.png'

        txt_path = BASEDIR + '/UALD/universal_landmark_detection/.eval/.._runs_GU2Net_runs_results_single_epoch000/chest/images/' + file_name + '.txt'
        with open(txt_path, 'r') as f:
            angles = f.readlines()



        result_old_path = BASEDIR + '/UALD/universal_landmark_detection/.eval/.._runs_GU2Net_runs_results_single_epoch000/chest/images/'
        result_new_path = BASEDIR + '/static/images/'
        shutil.copy(result_old_path + img_with_dots, result_new_path + img_with_dots)

        angle_name = ['CE角', '臼顶倾斜角','Sharp角','头臼指数']
        reference = ['20°-40°','<10°','≤40°','>75']
        left = angles[:4]
        right = angles[4:]
        for i in range(len(left)-1):
            left[i] = left[i] + '°'
        for i in range(len(right)-1):
            right[i] = right[i] + '°'
        angle_value = zip(angle_name,reference,left,right)

        return render_template('image.html', imgname=imgname, img_with_dots=img_with_dots, angle_value = angle_value)
        # return render_template('image.html', imgname=imgname, emotion=emotion, emotion_conf=emotion_conf,
        #                        img_with_dots=img_with_dots)


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


@app.route('/upload_video', methods=['POST'])
def upload_video():
    if request.method == 'POST':
        filepath = ''
        print(request.files)
        f = request.files['video']
        if f:
            imgname = secure_filename(f.filename)
            filepath = os.path.join(app.config['UPLOAD_PATH'], imgname)
            print('filepath:{}'.format(filepath))
            f.save(filepath)

            return render_template("video_file.html", imgname=imgname)
        return render_template("video_file.html")


if __name__ == '__main__':
    # DEBUG is SET to TRUE. CHANGE FOR PROD
    app.run(port=5000, debug=True)
