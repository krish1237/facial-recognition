#%%
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Input, Lambda, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop

# %%
def build_base_network(input_shape):

    seq = Sequential()

    nb_filter = [32,64]
    kernel_size = [5,3] 
    strides = 2

    seq.add(Conv2D(nb_filter[0], kernel_size[0], strides, input_shape=input_shape, padding="valid", data_format="channels_first"))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2,2),data_format="channels_first"))
    seq.add(Dropout(.25))

    seq.add(Conv2D(nb_filter[1], kernel_size[1], strides, padding="valid", data_format="channels_first"))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2,2),data_format="channels_first"))
    seq.add(Dropout(.25))

    seq.add(Flatten())
    seq.add(Dense(128, activation= "relu"))
    seq.add(Dropout(0.1))
    seq.add(Dense(50, activation= "relu"))
    return seq

#%%
X = np.load("siamese_data/X.npy")
Y = np.load("siamese_data/Y.npy")
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = .25)

#%%
input_dim = x_train.shape[2:]
img_a = Input(shape=input_dim)
img_b = Input(shape=input_dim)

#%%
base_network = build_base_network(input_dim)
feat_vecs_a = base_network(img_a)
feat_vecs_b = base_network(img_b)

#%%
def eucledian_distance(vects):
    x = vects[0]
    y = vects[1]
    return K.sqrt(K.sum(K.square(x-y), axis=1, keepdims = True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)

#%%
distance = Lambda(eucledian_distance, output_shape=eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])

epochs = 5
rms = RMSprop()

model = Model(inputs=[img_a, img_b], outputs = distance)
#%%
def contrastive_loss(y_true, y_pred):
    return K.mean(y_true*K.square(y_pred) + (1-y_true)*K.square(K.maximum(1 - y_pred, 0)))

#%%
model.compile(loss=contrastive_loss, optimizer=rms)

#%%
img1 = x_train[:,0]
img2 = x_train[:,1]

#%%
model.fit([img1,img2], y_train, validation_split = .25, batch_size = 128, verbose = 2, epochs = epochs)

# %%
pred = model.predict([x_test[:, 0], x_test[:, 1]])

#%%
def post_process(pred):
    pred = pred.ravel()
    pred = K.maximum(1-pred,0)
    pred = np.array(pred)
    return pred

# %%
pred = post_process(pred)

#%%
def save_model(model,filepath):
    model_json = model.to_json()
    with open(filepath+"/model.json","w") as f:
        f.write(model_json)
    model.save_weights(filepath+"/model_weights.h5")

#%%
save_model(model,"siamese_data")


# %%
