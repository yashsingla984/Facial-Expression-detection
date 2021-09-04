import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
X = np.genfromtxt('../input/ckpluspn/CKPlusN.csv', delimiter=',')
Y= X[:, 2304]
X=X[:, :2304]
X=X.reshape(X.shape[0], 48, 48)
def histogramNormalization(img):
    cdf=np.zeros(256)
    hist=np.zeros(256)
    for i in range(256):
        hist[i]=np.sum(img==i)
    hist=hist/np.sum(hist)
    cdf[0]=hist[0]
    for i in range(1, 256):
        cdf[i]=cdf[i-1]+hist[i]
    Ndis=norm.ppf(0.999*cdf)
    img=np.array(np.copy(img), dtype='float32')
    for i in range(256):
        img[img==i]=Ndis[i]
    return img
Xn=np.zeros(X.shape)
for i in range(X.shape[0]):
    #plt.imshow(np.array(X[i, :, :]*255/np.max(X[i, :, :]), dtype='uint8'), cmap='gray')
    #plt.show()
    #plt.imshow(histogramNormalization(np.array(X[i, :, :]*255/np.max(X[i, :, :]), dtype='uint8')), cmap='gray')
    #plt.show()
    Xn[i, :]=histogramNormalization(np.array(X[i, :, :]*255/np.max(X[i, :, :]), dtype='uint8'))
    print(i)
print(Xn.shape)
np.savetxt('/kaggle/working/FERFinalN.csv', np.concatenate((Xn.reshape(Xn.shape[0], 2304), Y[:, np.newaxis]), axis=1))