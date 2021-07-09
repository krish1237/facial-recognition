#%%
from tensorflow.keras.models import model_from_json
import faceDetection
import numpy as np
from numpy import expand_dims
from tensorflow.keras import backend as K
#%%
def load_model(filepath):
    json_file = open(filepath+"/model.json", 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(filepath+"/model_weights.h5")
    return model

#%%
model = load_model("siamese_data")

#%%
print(model.inputs)
print(model.outputs)

#%%
def get_face(required_size = (92,112)):
    faces = faceDetection.get_faces(required_size)
    face_pixels = faces[0].astype('float32')
    samples = expand_dims(face_pixels, axis=0)
    print(len(faces),faces[0].shape)
    return samples
#%%
def preprocess(img1,img2, size):
    img1 = img1[:,::size,::size]/255
    img2 = img2[:,::size,::size]/255
    img1 = expand_dims(img1, axis = 0)
    img2 = expand_dims(img2, axis = 0)
    print(img1.shape)
    print(img2.shape)
    return [img1,img2]
#%%
img1 = get_face()
img2 = get_face()
#%%
result = model.predict(preprocess(img1,img2,1))
# %%
def post_process(pred):
    pred = pred.ravel()
    pred = K.maximum(1-pred,0)
    pred = np.array(pred)
    return pred
#%%
result = post_process(result)

# %%
