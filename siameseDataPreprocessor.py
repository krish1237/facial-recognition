#%%
import pdb
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#%%
def read_image(filename):
    img = Image.open(filename)
    imgarr = np.array(img)
    return imgarr
#%%
img = read_image("dataset/s1/1.pgm")
plt.imshow(img, cmap="gray")
print(img.shape)

#%%
img = img[::2,::2]
plt.imshow(img, cmap="gray")

#%%
size = 1
total_sample_size = 10000

#%%
def get_data(size, total_sample_size):
    img = read_image('dataset/s'+str(1)+'/'+str(1)+'.pgm')
    img = img[::size,::size]
    dim1 = img.shape[0]
    dim2 = img.shape[1]

    count = 0

    x_genuine_pair = np.zeros([total_sample_size,2,1,dim1,dim2])
    y_genuine = np.ones([total_sample_size,1])

    for i in range(40):
        for j in range(int(total_sample_size/40)):
            ind1 = 0
            ind2 = 0

            while ind1 == ind2:
                ind1 = np.random.randint(10)
                ind2 = np.random.randint(10)
            img1 = read_image('dataset/s'+str(i+1)+'/'+str(ind1+1)+'.pgm')
            img2 = read_image('dataset/s'+str(i+1)+'/'+str(ind2+1)+'.pgm')

            img1 = img1[::size,::size]
            img2 = img2[::size,::size]

            x_genuine_pair[count, 0, 0, :, :] = img1
            x_genuine_pair[count, 1, 0, :, :] = img2
            count += 1
    
    count = 0

    x_imposite_pair = np.zeros([total_sample_size,2,1,dim1,dim2])
    y_imposite = np.zeros([total_sample_size,1])

    for i in range(int(total_sample_size/10)):
        for j in range(10):
            ind1 = 0
            ind2 = 0
            while ind1 == ind2:
                ind1 = np.random.randint(10)
                ind2 = np.random.randint(10)
            img1 = read_image('dataset/s'+str(ind1+1)+'/'+str(j+1)+'.pgm')
            img2 = read_image('dataset/s'+str(ind2+1)+'/'+str(j+1)+'.pgm')

            img1 = img1[::size,::size]
            img2 = img2[::size,::size]

            x_imposite_pair[count, 0, 0, :, :] = img1
            x_imposite_pair[count, 1, 0, :, :] = img2
            count += 1

    X = np.concatenate([x_genuine_pair,x_imposite_pair], axis = 0)/255
    Y = np.concatenate([y_genuine,y_imposite], axis = 0)

    return X,Y

#%%
X,Y = get_data(size,total_sample_size)

# %%
print(X.shape)
print(Y.shape)

# %%
np.save("siamese_data/X.npy",X)
np.save("siamese_data/Y.npy",Y)