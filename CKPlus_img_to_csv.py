#final CKplus
import glob
import numpy as np
from matplotlib import pyplot as plt
files=glob.glob('/kaggle/input/ckplus/CK+48/anger/*.png')
#print(files)
flag=True
y=0
for f in files:
    img=plt.imread(f)
    img=np.array(np.round(img*255.0/np.max(img), decimals=0), dtype='uint8')
    if flag:
        X=img.flatten()[np.newaxis, :]
        flag=False
    else:
        X=np.concatenate((X, img.flatten()[np.newaxis, :]), axis=0)
Y=np.repeat(y, len(files))
print(X.shape)
files=glob.glob('/kaggle/input/ckplus/CK+48/disgust/*.png')
y=1
for f in files:
    img=plt.imread(f)
    img=np.array(np.round(img*255.0/np.max(img), decimals=0), dtype='uint8')
    X=np.concatenate((X, img.flatten()[np.newaxis, :]), axis=0)
Y=np.concatenate((Y, np.repeat(y, len(files))))
print(X.shape)
files=glob.glob('/kaggle/input/ckplus/CK+48/fear/*.png')
y=2
for f in files:
    img=plt.imread(f)
    img=np.array(np.round(img*255.0/np.max(img), decimals=0), dtype='uint8')
    X=np.concatenate((X, img.flatten()[np.newaxis, :]), axis=0)
Y=np.concatenate((Y, np.repeat(y, len(files))))
print(X.shape)
files=glob.glob('/kaggle/input/ckplus/CK+48/happy/*.png')
y=3
for f in files:
    img=plt.imread(f)
    img=np.array(np.round(img*255.0/np.max(img), decimals=0), dtype='uint8')
    X=np.concatenate((X, img.flatten()[np.newaxis, :]), axis=0)
Y=np.concatenate((Y, np.repeat(y, len(files))))
print(X.shape)
files=glob.glob('/kaggle/input/ckplus/CK+48/sadness/*.png')
y=4
for f in files:
    img=plt.imread(f)
    img=np.array(np.round(img*255.0/np.max(img), decimals=0), dtype='uint8')
    X=np.concatenate((X, img.flatten()[np.newaxis, :]), axis=0)
Y=np.concatenate((Y, np.repeat(y, len(files))))
print(X.shape)
files=glob.glob('/kaggle/input/ckplus/CK+48/surprise/*.png')
y=5
for f in files:
    img=plt.imread(f)
    img=np.array(np.round(img*255.0/np.max(img), decimals=0), dtype='uint8')
    X=np.concatenate((X, img.flatten()[np.newaxis, :]), axis=0)
Y=np.concatenate((Y, np.repeat(y, len(files))))
print(X.shape)
files=glob.glob('/kaggle/input/ckplus/CK+48/contempt/*.png')
y=6
for f in files:
    img=plt.imread(f)
    img=np.array(np.round(img*255.0/np.max(img), decimals=0), dtype='uint8')
    X=np.concatenate((X, img.flatten()[np.newaxis, :]), axis=0)
Y=np.concatenate((Y, np.repeat(y, len(files))))
print(X.shape)
df=np.concatenate((X, Y[:, np.newaxis]), axis=1)
np.random.shuffle(df)
np.random.shuffle(df)
np.random.shuffle(df)
np.random.shuffle(df)
np.savetxt("CKplus.csv", df, delimiter=",")
print(df.shape)