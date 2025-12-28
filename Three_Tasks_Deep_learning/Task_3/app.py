import cv2
import os
from flask import Flask, request, render_template
from datetime import date
from datetime import datetime
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
import warnings
warnings.filterwarnings('ignore')





app = Flask(__name__)

MODEL_PATH = "static/face_recognition_model.h5"
LABEL_PATH = "static/labels.npy"
model = None
inv_label_map = {}
nimgs = 250

imgBackground=cv2.imread("background.png")

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []



def identify_face(image_path):
    
    model = load_model(MODEL_PATH)
    d_img = load_img(image_path, target_size=(100, 100))
    img = np.array(d_img)
    img = img / 255.0 # normalize the image
    img = img.reshape(1, 100, 100, 3) # reshape for prediction
    preds = model.predict(img)
       
    index = np.argmax(preds)          
    label = inv_label_map[index]      # label from mapping
    
    if os.path.exists(image_path):
        os.remove(image_path)
    
    return label


def train_model():
    
    global model, inv_label_map
    
    # faces = []
    # userlist = os.listdir('static/faces')
    # for user in userlist:
    #         for imgname in os.listdir(f'static/faces/{user}'):
    #             img = cv2.imread(f'static/faces/{user}/{imgname}')
    #             resized_face = cv2.resize(img, (100, 100))
    #             faces.append(resized_face.ravel())
    # faces = np.array(faces)
 
    
    train_generator = ImageDataGenerator(
    rescale=1./255,

    rotation_range=10,           
    width_shift_range=0.08,      # zyada move na ho
    height_shift_range=0.08,
    zoom_range=0.10,             # face proportion safe
    horizontal_flip=True,        # mirror faces OK

    brightness_range=[0.9, 1.1], # zyada extreme nahi
    shear_range=0.0,             # distortion se bachao
    channel_shift_range=0.0,     # skin tone bigar sakta hai
    fill_mode='nearest'
    )
    
    val_generator = ImageDataGenerator(
    rescale=1./255
    )
    
    train_iterator = train_generator.flow_from_directory(
    "static/faces/",
    target_size=(100,100),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
    )

    val_iterator = val_generator.flow_from_directory(
    "static/faces/",
    target_size=(100,100),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
    )
        
        
    label_map = train_iterator.class_indices
    inv_label_map = {v: k for k, v in label_map.items()}
    
    if os.path.exists(LABEL_PATH):
        os.remove(LABEL_PATH)
        np.save(LABEL_PATH, inv_label_map)
    else:
        np.save(LABEL_PATH, inv_label_map)  
   
    # early_stop = EarlyStopping(
    # monitor='val_loss',      # validation loss ko monitor kare
    # patience=3,              # 3 epochs tak improvement na ho
    # restore_best_weights=True
    # )
    
    # checkpoint = ModelCheckpoint(
    # "best_face_model.h5",
    # monitor='val_loss',
    # save_best_only=True
    # )
    
    model = Sequential([
        
                    Conv2D(16, (3,3), activation='relu', input_shape=(100,100,3)),
                    MaxPooling2D((2,2)),
                    Conv2D(64, (3,3), activation='relu'),
                    MaxPooling2D((2,2)),
                    Conv2D(64, (3,3), activation='relu'),
                    MaxPooling2D((2,2)),
                    Flatten(),
                    Dense(256, activation='relu'),
                    Dense(len(inv_label_map), activation='softmax')
    ])
    

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_iterator, epochs=10, validation_data=val_iterator)
    
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        model.save(MODEL_PATH)
    else:
        model.save(MODEL_PATH)

    
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')

def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l


@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/start', methods=['GET'])
def start():
    
    if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_PATH):
        global model, inv_label_map
        model = load_model(MODEL_PATH)
        inv_label_map = np.load(LABEL_PATH, allow_pickle=True).item()
    names, rolls, times, l = extract_attendance()

    # if 'face_recognition_model.h5' not in os.listdir('static'):
    #     return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (100, 100))
            
            img_name = 'pic.jpg'
            cv2.imwrite(img_name, frame[y:y+h, x:x+w])
            
            identified_person = identify_face(img_name)
            add_attendance(identified_person)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
            cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
            cv2.putText(frame, f'{identified_person}', (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
        imgBackground[162:162 + 480, 55:55 + 640] = frame
        cv2.imshow('Attendance', imgBackground)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)



@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs*5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

if __name__ == '__main__':
    app.run(debug=True)
