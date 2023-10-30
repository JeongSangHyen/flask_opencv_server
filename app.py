from flask import Flask, render_template,send_file, jsonify,Response
import cv2
import dlib
from scipy.spatial import distance
import time
import pyaudio
import wave
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

frame = None

def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A + B) / (2.0 * C)
    return ear_aspect_ratio

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def counter(func):
    def wrapper(*args, **kwargs):
        wrapper.count += 1
        time.sleep(0.05)
        global lastsave
        if time.time() - lastsave > 5:
            lastsave = time.time()
            wrapper.count = 0
        return func(*args, **kwargs)
    wrapper.count = 0
    return wrapper

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
WAVE_FILENAME = "wa.wav"

p = pyaudio.PyAudio()
wf = wave.open(WAVE_FILENAME, 'rb')
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

def play_alarm():
    wf = wave.open(WAVE_FILENAME, 'rb')
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(CHUNK)
    while data:
        stream.write(data)
        data = wf.readframes(CHUNK)

    stream.stop_stream()
    stream.close()
    wf.close()

@counter
def close():
    global lastsave
    
    print(frame)
    cv2.putText(frame, "DROWSY", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)

lastsave = time.time()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_image', methods=['POST'])
def get_image():
    image_path = 'static/drowst_image.jpg'  # 이미지 파일의 경로
    return send_file(image_path, mimetype='image/jpeg')

@app.route('/signup',methods=['GET','POST'])
def signup_page():
    return render_template('index2.html')

@app.route('/video_feed')
def video_feed():
    
    def generate():
        while True:
            _, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = hog_face_detector(gray)
            for face in faces:

                face_landmarks = dlib_facelandmark(gray, face)
                leftEye = []
                rightEye = []

                for n in range(36, 42):
                    x = face_landmarks.part(n).x
                    y = face_landmarks.part(n).y
                    leftEye.append((x, y))
                    next_point = n + 1
                    if n == 41:
                        next_point = 36
                    x2 = face_landmarks.part(next_point).x
                    y2 = face_landmarks.part(next_point).y
                    cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

                for n in range(42, 48):
                    x = face_landmarks.part(n).x
                    y = face_landmarks.part(n).y
                    rightEye.append((x, y))
                    next_point = n + 1
                    if n == 47:
                        next_point = 42
                    x2 = face_landmarks.part(next_point).x
                    y2 = face_landmarks.part(next_point).y
                    cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

                left_ear = calculate_EAR(leftEye)
                right_ear = calculate_EAR(rightEye)

                EAR = (left_ear + right_ear) / 2
                EAR = round(EAR, 2)

                if EAR < 0.15:
                    close()
                    print(f'close count : {close.count}')
                    if close.count == 15:
                        print("Driver is sleeping")
                        play_alarm()
                        save_path='static/drowst_image.jpg'
                        print(f'Image saved: {save_path}')
                        cv2.imwrite(save_path, frame)
                print(EAR)

            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)


cap.release()
cv2.destroyAllWindows()
p.terminate()