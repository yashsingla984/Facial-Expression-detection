# Facial-Expression-detection
A Convolution Neural Network (CNN) based model for facial expression recognition

                      Description
In this project we develop a Convolution Neural Network (CNN) based model for facial
expression recognition. Facial Expression Recognition is considered as a hard job. Even
human beings can get confused while classifying emotions. Also, it is possible that sometimes the given expression does not fit into any category of emotions that our model has.
Here, we present our algorithm in two parts. The first part consists of pre-processing
the images. The second part involves feeding the the pre-processed image into our CNN
based model. At the end, we evaluate our algorithm on various publicly available data
sets.

    Pre-processing the image
2.1 Face detection and cropping:
Given an image, we first need to convert the image into gray scale. Then we need to
identify the face and crop the area the containing face. The cropped image then needs to
be resized into shape of 48 x 48.
.....Immage.....

2.2 Photometric Normalization:
Different images are captured in different lightning conditions. Different regions of same
image can have different illumination. This difference in illumination can affect the performance of our CNN model. Here, we use homomorphic filtering based normalisation
mentioned in [3]. Homomorphic filtering assumes an image, F(x, y), as product of two
quantities- illumination , I(x, y), and reflectance, R(x, y).
We used INface toolbox on Matlab for homomorphic filtering based photo normalization.
Images after photometric normalization are shown below. One can see that variation of
intensity within images has been flattened.
...............Image.......

2.3 Histogram matching:
Images are often taken in different lightning conditions due to which the intensity value
of pixel and their distribution differs from one image to another. This can lead to a
neural network learning useless features in image which has nothing to do with facial
expressions. Neural network tend to perform better if the data is normalised and confined
2
to small range. For this purpose we matched the histogram of our image to a standard
normal distribution (µ = 0 and σ = 1). This makes illumination of different images
same. Examples of images are shown below where histogram matched images have been
re-scaled for plotting.

3.   Architecture of our model
....Image.....
The construction of our CNN model is shown above. Apart from regular convolution layers and max pooling layer we have added dropout layers in between. This prevents our
model from over fitting and helps it to learn better features. The model was trained using
the pre-processed images obtained by methods mentioned in Section 2

4.Results
We tested our algorithms on two popularly used data sets- CK+ and FER-2013. CK+ is
a small data set containing 981 images. Human beings can easily see and classify CK+
images into various categories. We obtained an accuracy if 99.92% on this data set. 
FER2013 is a large data set with over 42,000 images. Human beings face difficulty in classifying
FER-2013 images. Since, this data set is quite large so we used a smaller version of it,
called FER13 Cleaned Dataset, which is available on Kaggle. FER13 Cleaned Dataset
contains about 16,900 images. We obtained an accuracy of 72% on it. These results are
comparable to accuracy of human beings. We also outperformed the model in [5].

5 Conclusion
The model obtained by us is relatively simple compared to some of the other existing
models, however, the accuracy obtained is comparable to these models. The reason of
obtaining good accuracy using a simple model is the pre-processing steps involved in our
algorithm. Also, the introduction of dropout layers within the CNN structure allowed us
to obtain higher validation accuracy. We were able to obtain accuracy levels of a normal
human being on these datasets. These datasets are actually quite unbalanced. A balanced
dataset would have provided us a better trained model.
