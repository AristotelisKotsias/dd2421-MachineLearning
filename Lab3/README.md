# **Bayes Classifiers and Boosting**
Implementation of a Bayes Classifier and the Adaboost algorithm that improves the performance of a weak classifier by aggregating multiple hypotheses generated across different distributions of the training data. Three datasets for classification are been used where each dataset is provided in the form of two csv files: datasetnameX.txt for the feature vectors and datasetnameY.txt for the corresponding labels.

**Datasets:**
- **Iris**: This dataset contains 150 instances of 3 different types of iris plants. The feature consists of 2 attributes describing the characteristics of the Iris. The task is to classify instances as belonging to one of the types of Irises.
- **Vowel**. This dataset contains 528 instances of utterances of 11 different vowels. The task is to classify instances as belonging to one of the types of the vowels.
- **Olivetti Faces**. This dataset contains 400 different images of the faces of 40 different persons, each person is represented by 10 images. The images were taken at different times with varying lighting and facial expressions against a dark homogeneous background. They are all grayscale, and 64x64 pixels. The task is to classify instances as belonging to one of the people.


**Packages used:**
- [x] numpy
- [x] scipy
- [x] matplotlib
- [x] sklearn
