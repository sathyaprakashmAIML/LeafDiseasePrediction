from keras.preprocessing import image
from keras.models import model_from_json
import numpy as np
import os

json_file=open('model.json','r')
loaded_model=json_file.read()
json_file.close()
model=model_from_json(loaded_model)
model.load_weights('model.weights.h5')
print('model load')

def classify(file):
    test_img=image.load_img(file,target_size=(128,128,3))
    test_img=image.img_to_array(test_img)
    test_img=np.expand_dims(test_img,axis=0)
    result=model.predict(test_img)
    print(result)
    fresult=np.max(result)
    label=["Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___Healthy",
       "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_",
       "Corn_(maize)___Healthy","Corn_(maize)___Northern_Leaf_Blight","Grape___Black_rot",
       "Grape___Esca_(Black_Measles)","Grape___Healthy","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
       "Potato___Early_blight","Potato___Healthy","Potato___Late_blight","Tomato___Bacterial_spot",
       "Tomato___Early_blight","Tomato___Healthy","Tomato___Late_blight","Tomato___Leaf_Mold",
       "Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot",
       "Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus"]
    label2=label[result.argmax()]
    print(label2)
    
files=[]
path=r'D:\ML\cv2\deep\leaf disease\dataset\check'
for r,d,f in os.walk(path):
    for file in f:
        if file.endswith('.JPG'):
            files.append(os.path.join(r,file))

for f in files:
    classify(f)
    print('/n')

