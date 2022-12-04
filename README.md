

In this project, an attempt is made to deep-dive into the ‘Vinho Verde’ wine dataset, better understand it using exploratory data analysis and classify and predict the quality of the ‘Vinho Verde’ wine samples, for both the red and white wine types, using four different machine learning algorithms, namely: decision tree, random forest, k-nearest neighbour (k-NN) and support vector machine (SVM).  

The data used, for our analysis, is the ‘Vinho Verde’ wine samples available on the UCI, Machine Learning Repository, 
http://archive.ics.uci.edu/ml/datasets/Wine+Quality. This wine quality dataset contains physiochemical information related to both the red and white wine variants of the Portuguese ‘Vinho Verde’ wine, obtained from the Minho (northwest) region of Portugal. This dataset was donated by Paulo Cortez, Full Professor at the Department of Information Systems, School of Engineering, University of Minho, Portugal to the UCI Data Repository on October 07, 2009. 

This dataset has a total of twelve attributes. There are a total of 4898 samples of the white wine variant and 1599 samples of the red wine variant. Both the red and white wine samples are qualified by a set of eleven physicochemical features. These physicochemical attributes are: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates and alcohol content. All these eleven attributes would be continuously distributed and be of the ‘float’ datatype. The twelfth attribute, the quality attribute, can be considered as a sensory information indicator, indicating a quality score between 0 and 10 and would be of the ‘integer’ datatype. This quality attribute is our attribute of interest, being attempted to classify and predict by applying the machine learning algorithms.

The approach used would be as follows:

Firstly, both the white wine (4898 samples) and red wine (1599 samples) samples are combined to get one consolidated wine sample dataset of 4898 + 1599 = 6497 entries. This consolidated dataset is the sample set used for our analysis.  

Next, detailed exploratory data analysis is performed to understand the individual distributions of all the parameters, how they interact with respect to the other attributes, their inter-dependencies and try to find patterns, if any and review the correlations between the different set of features. A check for missing values and other outlier data is also performed and appropriately treated. 

Our attribute of interest, the 'quality' attribute, is then divided into four distinct quality labels: low, medium, high and excellent.
These four levels are: Level 1: Low (for quality attribute values: 0, 1 and 2) ; Level 2: Medium (for quality attribute values: 3, 4 and 5) ; Level 3: High (for quality attribute values: 6, 7 and 8) ; Level 4: Excellent (for quality attribute values: 9 and 10).

Since in our case, all the attributes follow different distributions and have different ranges and scale, feature scaling becomes an important task to perform. Feature scaling, also known as data normalization or standardization, essentially enables all the selected attributes to lie within the same scale. Since all the attributes, after normalization would be between the same intervals, they all have the same importance in our analysis. The other advantage is, the classifier algorithms we are planning to implement, produces more consistent results on normalized data, and hence it is considered a good practice to perform data normalization before performing the classification tasks. There are different methods to normalize data and we are using the Min-Max normalization technique for our standardization. 

Also, used the 'analysis of variance', ANOVA method, to understand the physicochemical features, impacting the 'quality' attribute the most and to perform feature selection. Based on the results, we see that the 'alcohol', 'density', 'volatile acidity' and 'chlorides' attributes would be impacting quality the most. These results are also consistent with the results of the correlation matrix. The other interesting thing, is that the scores of the other seven physicochemical features, would be very low as compared to these four attributes, and we can omit the lower score features and use a subset of these four high score features for our predictive model building tasks. 

The next step is to split the dataset into two subsets, a training set and a test set. We are using three variations of train: test splits, i.e. 80:20, 75:25 and 70:30, to limit the impact of inherent biases. 

Finally, we are implementing the four different machine learning algorithms namely: Decision Tree, Random Forest, k-Nearest Neighbour (k-NN) and Support Vector Machine (SVM), first by using all the attributes in the dataset and then by using the subset of the four most-relevant selected features ('alcohol', 'density', 'volatile acidity' and 'chlorides'), for all the three train: test variations (80:20, 75:25, 70:30), in an attempt to best classify and predict the quality attribute of this wine dataset.


After reviewing the model performance of all the implemented iterations, we see that the accuracy scores, when using the full set of features and when using the subset of the four selected features, is almost the same, for all the four algorithms. This would be consistent with our ANOVA (analysis of variance) findings, where we see that the scores of the other seven physicochemical features is very low as compared to these four features, hence a much lower impact on the ‘quality’ attribute. 

The accuracy scores would be between the ranges, 72.820% and 83.630%. The lowest, 72.820%, for the Decision Tree implementation, using the subset of four selected features, for the 70:30, train: test split and the highest, 83.630%, for the Random Forest implementation, using all features in the sample set, for the 75:25, train: test split.

All the four implemented algorithms provide robust performance. Amongst them, the Random Forest algorithm, using all features, gives us the best overall performance for all the three train: test variations.


The computational work has been performed using Python 3 (ipykernel) on the Jupyter Notebook (anaconda3) platform and the important Python libraries and modules used, would be: Pandas, NumPy, Matplotlib, Seaborn and Scikit-Learn.

A summarization of the findings and the Jupyter, .ipynb file, which would have the actual codes, is shared in this repository for reference.  

