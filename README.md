# Calorie-Predictor
Team Members: Yiming Miao, Xiaoyu An, Tingyi Zhang
Contacts: {mym1031, anxiaoyu, tingyi97}@bu.edu

**Check this out!**
<img src="https://github.com/Yiming-Miao/Calorie-Predictor/blob/master/A1_11.jpg">

## Things we established
### Website
<img src="https://github.com/Yiming-Miao/Calorie-Predictor/blob/master/Website.png">

### Food Recognize Algorithm
- The deep learning model we used is Mask R-CNN.  
- Our work is based on a GitHub repo of Matterport, please click this link to see their excellent work: https://github.com/matterport/Mask_RCNN.

### Our Own Dataset
We built a relatively small dataset, manually labeled by VIA(VGG Image Anotator) and containing 10 types of foods.

## Product Mission
Use machine learning to recognize food images and estimate weights, and give prediction of its calories. Calorie Predictior doesn't need users to know the weight of foods or enter names of foods, all it needs is a picture.
<img src="https://github.com/Yiming-Miao/Calorie-Predictor/blob/master/productmission.png">

## Target Users
Everyone who try to calculate how much they eat

## User Stories
- I, the college student who try to keep on diet in order to lose weights, but I don't know the name of foods and weights of foods I eat in the canteen, so I need a tool to help me analyze my foods.
- I, the professional bodybuilder, who try to keep very strict diet before my show, and I want to know exactly calories I intake everyday.
- I, the patient, who have health problem needed to avoid eating high calorie foods, need to know the energy of foods before taking them.

## MVP
- An user interface that allow users to input foods images and enter the weight of foods
- Capability of reconizing foods
- Calculate calories based on calorie table

## System Design (Original idea)
<img src="https://github.com/Yiming-Miao/Calorie-Predictor/blob/master/system.png">

### Pogramming Language
#### Deep Learning Algorthm
Python 
#### Web
Our web page is designed based on Tornado (python web framework), HTML (Materialize framework) and JavaScript.

## Similar Products Analysis  
We have found several apps in Apple Store that are related to calorie predicting.

**Lose it!**  
https://www.loseit.com/

<img src="https://github.com/Yiming-Miao/Calorie-Predictor/blob/master/loseit.png">

Pros:  
- In this app, the users could enter their weight and height. Then users could set a goal of how much weight they are going to lose. However, the users cannot upload their food images in order to get the food's calorie predicted value automaticly.

Cons:  
- This app only allows the users to upload their food images, choose the types of food and add the grams of food. Therefore, the main purpose of this app is just recording users' everyday calorie intake manually.  

**Calorie Mama**  
https://caloriemama.ai/

<img src="https://github.com/Yiming-Miao/Calorie-Predictor/blob/master/caloriemama.png">
Pros: 
- It can classify the food type automatically.

Cons: 
- It needs manually inputing the weight of the food.

**Myfitnesspal**  
https://www.myfitnesspal.com/

<img src="https://github.com/Yiming-Miao/Calorie-Predictor/blob/master/myfitnesspal.png">

Pros:  
- The advantage of this app is that it can give the users a more specific nutrition composition of the food, such as protain content, carbohydrate content, fat content and so on. Moreover, it can give suggestions about the amount of calorie intaking to the users in view of thier physical data and weight lose goal.

Cons:  
- This app cannot identify pictures of food. The users should enter the food's name and weight. The advantage of this app is that it can give the users a more specific nutrition composition of the food, such as protain content, carbohydrate content, fat content and so on. Moreover, it can give suggestions about the amount of calorie intaking to the users in view of thier physical data and weight lose goal. 

### Classification Technology Justifications
**Random Forests**

Random forest generation method are described in four steps 
1. N samples are generated from the sample set by resampling.
2. If we set the number of sample features is A, and K features in A are selected from N samples, and the optimal segmentation point is obtained by establishing a decision tree.
3. Repeat M times to generate M decision trees.
4. Majority voting mechanisms make predictions.

Pros:
- Random forests can process very high-dimensional data (that is, data for many features) without the need for feature selection.
- After the training, what characteristics is more important can be given by the random forest.
- The training speed is fast and it is easy to make a parallelization method (when training, the tree and the tree are independent of each other).
- During the training process, the effects between the features can be detected.
- If a large part of the features are lost, the accuracy can still be maintained with the RF algorithm.
- The random forest algorithm has strong anti-interference ability. So when there is a large amount of data missing in the data, it is good to use RF.

Cons:
- There may be many similar decision trees that mask the real results.
- For small data or low-dimensional data (data with less features), it may not produce a good classification. 

**Support Vector Machines (SMO version)**

Pros:
- Perform accurate in high dimensional spaces;
- Work efficiently with small data
- SMO use a subset of training points in the decision function (called support vectors), so it’s also memory efficient.

Cons:
- The algorithm is prone for over-fitting, if the number of features is much greater than the number of samples.
- SVMs do not directly provide probability estimates, which are desirable in most classification problems.
- SVMs are not very efficient computationally, if your dataset is very big, such as when you have more than one thousand rows.

**Naive Bayes**

Pros:
- When dealing with large datasets or low-budget hardware, Naive Bayes algorithm is a feasible choice for short training time.
- The prediction time of this algorithm is very efficient.
- Transparency. It is easy to understand which features are influencing the predictions. 

Cons:
- It has strong feature independence assumptions.
- The prior probability needs to be known, and the prior probability often depends on the hypothesis. There are many models that can be assumed, so in some cases the prediction effect will be poor due to the hypothetical prior model.
- Because we determine the classification by a priori and data to determine the probability of the posterior, there is a certain error rate in the classification decision.
- Very sensitive to the form of input data.

### Weight Estimate(TBD)
<img src="https://github.com/Yiming-Miao/Calorie-Predictor/blob/master/weightestimation1.png">

<img src="https://github.com/Yiming-Miao/Calorie-Predictor/blob/master/weightestimation2.png">

In our system, after segmentation and classification, the most important thing is describging or estimating the weight of different food. The MVP of this part is just let the user to enter the weight of the food, then we can calculate calories of a meal. If we want our system to be more intelligent, we should let our system learn to estimate the weight of the food in the image.

There are two ways to estimate the weight of food based on different food types. We assume that the food can be classified into two categories. One type of food is the kind that we can use some combinations of mathematical solid geometry to represent the shape of food in order to get thier volumes. Since we have classified the food, we can know the density information to calculate the weight of the food. This kind of technology is called the shape template 3D reconstruction estimating.

The other type of food is the kind that cannot use regular mathmetical solid geometry to represent. They might just occupy some area such as scrambled eggs. Since the food images are usually geometrically distorted, we need to convert the food images to some kind of front view images in order to estimate the food area. After the food in the image is identified and the area is estimated, we estimate the weight of the food using the area-weight relation in the training data of the corresponding food item. 

**Referrence:**
1. He Y, Xu C, Khanna N, Boushey CJ, Delp EJ. FOOD IMAGE ANALYSIS: SEGMENTATION, IDENTIFICATION AND WEIGHT ESTIMATION. Proc (IEEE Int Conf Multimed Expo). 2013;2013:10.1109/ICME.2013.6607548. doi:10.1109/ICME.2013.6607548  
2. Estrada, F.J. & Jepson, A.D. Int J Comput Vis (2009) 85: 167. https://doi.org/10.1007/s11263-009-0251-z  
3. Liang, Yanchao & Li, Jianhua. (2017). Computer vision-based food calorie estimation: dataset, method, and experiment. https://arxiv.org/pdf/1705.07632.pdf

### Available Opensource Dataset
- **MS-COCO**: http://cocodataset.org/#home  
<img src="https://github.com/Yiming-Miao/Calorie-Predictor/blob/master/mscocofig.png">

- **Food-101**：https://www.vision.ee.ethz.ch/datasets_extra/food-101/  
<img src="https://github.com/Yiming-Miao/Calorie-Predictor/blob/master/food101.png">

- **ECUSTFD**： https://github.com/Liang-yc/ECUSTFD-resized-  
<img src="https://github.com/Yiming-Miao/Calorie-Predictor/blob/master/ecustfig.png">

- **UNIMIB2015 Food Database， UNIMIB2016 Food Database**: http://www.ivl.disco.unimib.it/activities/food-recognition/  
<img src="https://github.com/Yiming-Miao/Calorie-Predictor/blob/master/unimibfig.png">

- **Calorie MAMA API**: https://dev.caloriemama.ai/  
<img src="https://github.com/Yiming-Miao/Calorie-Predictor/blob/master/apifig.png">


## Patent analysis  
**Food recognition:**
1. Deep learning-based food image identifying method:https://patents.google.com/patent/CN104636757A/en?oq=food+image+recognition

**Nutrition analysis:**  
1. System and method for nutrition analysis using food image recognition:https://patents.google.com/patent/US9349297?oq=food+image+recognition  
2. Dietary assessment system and method:https://patents.google.com/patent/US8605952?oq=food+image+recognition  
3. Automated food recognition and nutritional estimation with a personal mobile electronic device:https://patents.google.com/patent/US9734426?oq=foods+weight+recognition  
4. Restaurant-specific food logging from images: https://patents.google.com/patent/US9659225?oq=estimate+foods+weight+from+image

**Food volume estimation:**  
1. Method and apparatus to determine product weight and calculate price using a camera:https://patents.google.com/patent/US20050097064?oq=estimate+foods+weight+from+image  
2. Method for computing food volume in a method for analyzing food:https://patents.google.com/patent/US20110182477?oq=foods+volume+estimation
