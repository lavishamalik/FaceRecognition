import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask,jsonify,request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

app.config["IMAGE_UPLOADS"] = "/Users/lavisha/PythonProjects/datasetface"
app.config["TEST_UPLOADS"] = "/Users/lavisha/PythonProjects/testface"

@app.route("/face/save", methods=["POST"])
@cross_origin()
def save_data():
    face_cascade = cv2.CascadeClassifier(r"haarcascade_frontalface_alt.xml")
    dataset_path = '/Users/lavisha/PythonProjects/datasetface/'
    image = request.files["image"]
    image.save(os.path.join(app.config["IMAGE_UPLOADS"],image.filename))
    file_name = image.filename
    print(file_name)

    skip = 0
    face_data = []
    img = cv2.imread(dataset_path + image.filename) # From android 
    print(img)
    print(img.shape)
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame,1.3,5) #frame,scaling factor,number of neighbors
    print(faces)
    faces = sorted(faces, key = lambda f:f[2]*f[3]) #sort according to face area w*h
    for (x,y,w,h) in faces[-1:]: #consider largest face first
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        offset = 10
        face_section = img[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        face_data.append(face_section)
        print(len(face_data))

    face_data = np.asarray(face_data)
   # plt.imshow(face_data[0], cmap='gray')
    face_data = face_data.reshape((face_data.shape[0],-1))
    print(face_data.shape)
    np.save(dataset_path+file_name+'.npy',face_data)
    return "Data successfully saved"

# KNN
def distance(x1,x2):
    return np.sqrt(((x1-x2)**2).sum())

def knn(train, test,k=5):
    vals = []
    m = train.shape[0]
    for i in range(m):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test,ix)
        vals.append([d,iy])
    dk = sorted(vals, key = lambda x:x[0])[:k]
    labels = np.array(dk)[:,-1]
    new_vals = np.unique(labels, return_counts=True)
    index = np.argmax(new_vals[1])
    pred = new_vals[0][index]
    return pred

@app.route("/face/predict", methods=["POST"])
@cross_origin()
def predict():
    face_cascade = cv2.CascadeClassifier(r"haarcascade_frontalface_alt.xml")
    skip = 0

    print("Inside predict")

    face_data = []
    label = []

    class_id = 0      #labels for given file
    names = {}        #mapping between id-name

    dataset_path = '/Users/lavisha/PythonProjects/datasetface/'
    testface_path = '/Users/lavisha/PythonProjects/testface/'

    #Data Preparation
    for fx in os.listdir(dataset_path):
        if fx.endswith('.npy'):
            names[class_id] = fx[:-4] #Mapping between class label and output
            print("Loaded "+fx)
            data_item = np.load(dataset_path+fx)
            face_data.append(data_item)
            target = class_id*np.ones((data_item.shape[0],))
            class_id = class_id+1
            label.append(target)

    face_dataset = np.concatenate(face_data,axis=0)
    face_labels = np.concatenate(label, axis=0).reshape((-1,1))
    print(face_dataset.shape)
    print(face_labels.shape)

    trainset = np.concatenate((face_dataset, face_labels), axis=1)
    print(trainset.shape)

    image = request.files["image"]
    image.save(os.path.join(app.config["TEST_UPLOADS"],image.filename))

    skip = 0
    face_data = []

    frame = cv2.imread(testface_path + image.filename) # From android
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame,1.3,5) #frame,scaling factor,number of neighbors
    for face in faces:
        x,y,w,h = face
        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        out = knn(trainset, face_section.flatten())
        pred_name = names[int(out)]
        cv2.putText(frame, pred_name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255),2)
    #plt.imshow(frame, cmap='gray')
    return jsonify({"name":pred_name})

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)