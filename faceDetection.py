#%%
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from camera import capture
from PIL import Image
from numpy import asarray
#%%
detector = MTCNN()
#%%
def get_faces(required_size=(160,160)):
    pixels = capture()
    pixels = asarray(pixels)
    faces = detector.detect_faces(pixels)
    faces_array = list()
    i = 0
    for face in faces:
        x,y,width,height = face['box']
        x-=5
        y-=5
        width+=10
        height+=10
        face_img = pixels[y:y+height, x:x+width]
        plt.subplot(1, len(faces), i+1)
        i+=1
        face_img = Image.fromarray(face_img)
        face_img = face_img.resize(required_size)
        face_img = face_img.convert('L')
        face_img = asarray(face_img)
        plt.imshow(face_img, cmap='gray')
        faces_array.append(face_img)
    plt.show()
    return faces_array