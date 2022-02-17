# Face-Bot
- Gender and age detector with fun functionalities 
- The main aim of this article is to detect age and gender through the given data set. We will use simple python and Keras methods for detecting age and gender.


# About the Project :

- In this Python Project, I had used Deep Learning to accurately identify the gender and age of a person from a single image of a face. I used the models trained by <a href="https://talhassner.github.io/home/projects/Adience/Adience-data.html">Tal Hassner and Gil Levi</a>. The predicted gender may be one of ‘Male’ and ‘Female’, and the predicted age may be one of the following ranges- (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100) (8 nodes in the final softmax layer). It is very difficult to accurately guess an exact age from a single image because of factors like makeup, lighting, obstructions, and facial expressions. And so, I made this a classification problem instead of making it one of regression.</p>

- In Gender Detection we have train our own model with UTK Dataset

- In Addition You can Adjust Brightness , Contrast , Blurring by your own 😉

- Even You can find your friends eyes and make their cartoon images 👀

# Dataset 
<p>For this python project, I had used the UTKFace dataset; the dataset is available in the public domain and you can find it <a href="https://github.com/aicip/UTKFace">here</a>. 
</p><p>UTKFace dataset is a large-scale face dataset with long age span (range from 0 to 116 years old). The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity. The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc. This dataset could be used on a variety of tasks, e.g., face detection, age estimation, age progression/regression, landmark localization, etc. Some sample images are shown as following</p>
<br>

## What is Computer Vision?
- Computer Vision is the field of study that enables computers to see and identify digital images and videos as a human would. 
- The challenges it faces largely follow from the limited understanding of biological vision. 
- Computer Vision involves acquiring, processing, analyzing, and understanding digital images to extract high-dimensional data from the real world in order to generate symbolic or numerical information which can then be used to make decisions. The process often includes practices like object recognition, video tracking, motion estimation, and image restoration.
## What is OpenCV?
- OpenCV is short for Open Source Computer Vision. Intuitively by the name, it is an open-source Computer Vision and Machine Learning library. This library is capable of processing real-time image and video while also boasting analytical capabilities. It supports the Deep Learning frameworks TensorFlow, Caffe, and PyTorch.

## What is Deep Learning?
- Deep Learning involves taking large volumes of structured or unstructured data and using complex algorithms to train neural networks. It performs complex operations to extract hidden patterns and features (for instance, distinguishing the image of a cat from that of a dog).

Hear cascade link it [here ](https://github.com/opencv/opencv/blob/master/data/haarcascades) prtotxt and .caffemodel from this [link](https://talhassner.github.io/home/publication/2015_CVPR)

<br>

# Run Locally

Clone the project

```bash
  git clone https://github.com/ruchita-oza/Gender-And-Age-Detaction.git
```

Go to the project directory

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  streamlit run app.py
```


# Authors
- [@ruchita-oza](https://www.github.com/ruchita-oza)
- [@ninama889](https://www.github.com/ninama889)
