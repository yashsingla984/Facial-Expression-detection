

Smart Facial expression Recognition

System

Mohammad Atif

2017EE10462

Yash Singla

2017EE10505

Mudit Soni

2017EE10463

July 1, 2020

1 Description

In this project we develop a Convolution Neural Network (CNN) based model for facial

expression recognition. Facial Expression Recognition is considered as a hard job. Even

human beings can get confused while classifying emotions. Also, it is possible that some-

times the given expression does not ﬁt into any category of emotions that our model has.

Here, we present our algorithm in two parts. The ﬁrst part consists of pre-processing

the images. The second part involves feeding the the pre-processed image into our CNN

based model. At the end, we evaluate our algorithm on various publicly available data

sets.

2 Pre-processing the image

2.1 Face detection and cropping:

Given an image, we ﬁrst need to convert the image into gray scale. Then we need to

identify the face and crop the area the containing face. The cropped image then needs to

be resized into shape of 48 x 48.

Figure 1: Original images from CK+ data set

1





2.2 Photometric Normalization:

Diﬀerent images are captured in diﬀerent lightning conditions. Diﬀerent regions of same

image can have diﬀerent illumination. This diﬀerence in illumination can aﬀect the per-

formance of our CNN model. Here, we use homomorphic ﬁltering based normalisation

mentioned in [3]. Homomorphic ﬁltering assumes an image, F(x, y), as product of two

quantities- illumination , I(x, y), and reﬂectance, R(x, y).

F(x, y) = I(x, y)R(x, y)

Next step is to take logarithm of F(x, y) and the take its fourier transform.

log(F(x, y)) = log(R(x, y)) + log(I(x, y))

(1)

(2)

(3)

(4)

F(log(F(x, y))) = F(log(R(x, y))) + F(log(I(x, y)))

F (u, v) = R (u, v) + I (u, v)

0

0

0

0

0

0

where F (u, v), I (u, v) and R (u, v) are fourier transform of log(F(x, y)), log(R(x, y)) and

log(I(x, y)).

Next step is to apply a high pass ﬁlter to suppress low frequency components. Consider

a high pass ﬁlter, H(u, v) which has entries corresponding to high frequency component

as 1 and entries corresponding to low frequency component less than 1.

0

0

0

H(u, v)F (u, v) = H(u, v)R (u, v) + H(u, v)I (u, v)

(5)

Next we take inverse fourier transform and ﬁnally take exponential of the result to obtain

ﬁltered image, G(x, y).

G0(x, y) = F−1(H(u, v)F0(u, v))

G(x, y) = exp(G (x, y))

(6)

(7)

0

We used INface toolbox on Matlab for homomorphic ﬁltering based photo normalization.

Images after photometric normalization are shown below. One can see that variation of

intensity within images has been ﬂattened.

Figure 2: Images after Photometric Normalization (Intensity variation within images has been

ﬂattened)

2.3 Histogram matching:

Images are often taken in diﬀerent lightning conditions due to which the intensity value

of pixel and their distribution diﬀers from one image to another. This can lead to a

neural network learning useless features in image which has nothing to do with facial

expressions. Neural network tend to perform better if the data is normalised and conﬁned

2





to small range. For this purpose we matched the histogram of our image to a standard

normal distribution (µ = 0 and σ = 1). This makes illumination of diﬀerent images

same. Examples of images are shown below where histogram matched images have been

re-scaled for plotting.

Figure 3: Images after histogram matching (Illumination of the two images have become same

after histogram matching)

3 Architecture of our model

The construction of our CNN model is shown above. Apart from regular convolution lay-

ers and max pooling layer we have added dropout layers in between. This prevents our

model from over ﬁtting and helps it to learn better features. The model was trained using

the pre-processed images obtained by methods mentioned in Section 2.

3





4 Results

We tested our algorithms on two popularly used data sets- CK+ and FER-2013. CK+ is

a small data set containing 981 images. Human beings can easily see and classify CK+

images into various categories. We obtained an accuracy if 99.92% on this data set. FER-

2013 is a large data set with over 42,000 images. Human beings face diﬃculty in classifying

FER-2013 images. Since, this data set is quite large so we used a smaller version of it,

called FER13 Cleaned Dataset, which is available on Kaggle. FER13 Cleaned Dataset

contains about 16,900 images. We obtained an accuracy of 72% on it. These results are

comparable to accuracy of human beings. We also outperformed the model in [5].

Dataset

CK+

FER-2013 (Cleaned dataset)

Accuracy

99.92%

72%

Table 1: Validation accuracy on diﬀerent datasets

Confusion matrix for diﬀerent datasets are also shown below.

An Di Fe Ha Sa

Su

0

Co

0

An

Di

Fe

1

0

0

0

0

1

0

0

1

0

0

0

0

0

0

0

1

0

0

0

0

0

0

0

1

0

0

0

0

0

0

0

1

0

0

0

0

0

0

Ha

Sa

Su

Co

0

0

0

0.996 0.004

0

0

1

Table 2: Confusion matrix for CK+

An

An 0.624

Di 0.273 0.614

Di

Fe

0.140 0.054 0.182

0.023 0.091

Ha

Ne

0

0

Fe 0.196 0.014 0.512 0.056 0.221

Ha 0.036 0.001 0.038 0.808 0.117

Ne 0.071 0.002 0.052 0.116 0.759

Table 3: Confusion matrix for FER-2013 (Cleaned dataset)

5 Conclusion

The model obtained by us is relatively simple compared to some of the other existing

models, however, the accuracy obtained is comparable to these models. The reason of

obtaining good accuracy using a simple model is the pre-processing steps involved in our

algorithm. Also, the introduction of dropout layers within the CNN structure allowed us

to obtain higher validation accuracy. We were able to obtain accuracy levels of a normal

4





human being on these datasets. These datasets are actually quite unbalanced. A balanced

dataset would have provided us a better trained model.

6 References

\1. Ali Mollahosseini, Behzad Hasani, Mohammad H. Mahoor, ”AﬀectNet: A Database

for Facial Expression, Valence, and Arousal Computing in the Wild”

\2. Raviteja Vemulapalli and Aseem Agarwala, ”A Compact Embedding for Facial Ex-

pression Similarity”

\3. J. Short, J. Kittler and K. Messer, “A comparison of photometric normalisation algo-

rithms for face veriﬁcation,” in Sixth IEEE International Conference on Automatic Face

and Gesture Recognition, May 2004, pp. 254-159.

\4. K. Liu, M. Zhang and Z. Pan, ”Facial Expression Recognition with CNN Ensemble,”

2016 International Conference on Cyberworlds (CW), pp. 163-166, 2016

5


