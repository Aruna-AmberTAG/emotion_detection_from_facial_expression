from flask import Flask, render_template, request
import cv
import pickle
from keras.models import load_model
import numpy as np


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

@app.route('/')
def index():
    return render_template('index.html')


    

@app.route('/after', methods=['GET', 'POST'])
def after():
    img = request.files['file1']

    img.save('static/file.jpg')

    img1 = cv.imread('static/file.jpg')
    gray = cv.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    cascade = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    faces = cascade.detectMultiScale(gray, 1.1, 3)

    for x,y,w,h in faces:
        cv.rectangle(img1, (x,y), (x+w, y+h), (0,255,0), 2)

        cropped = img1[y:y+h, x:x+w]

    cv.imwrite('static/after.jpg', img1)

    try:
        cv.imwrite('static/cropped.jpg', cropped)

    except:
        pass

    try:
        image = cv.imread('static/cropped.jpg', 0)
    except:
        image = cv.imread('static/file.jpg', 0)
        
    image = cv.imread('static/file.jpg',0)
    print(image.shape)

    image = cv.resize(image,(48,48))
    image = image/255.0
    image = np.reshape(image, (1,48,48,1))

    model = load_model('model.h5')

    prediction = model.predict(image)

    label_map =   ['Anger','Neutral' , 'Fear', 'Happy', 'Sad', 'Surprise']

    prediction = np.argmax(prediction)

    final_prediction = label_map[prediction]

    return render_template('after.html', data = final_prediction)




if __name__ == "__main__":
	app.run(debug=True)
