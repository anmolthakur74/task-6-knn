# Task 6: K-Nearest Neighbors (KNN) Classification on Iris Dataset

This repository contains my solution for **Task 6** of the AI & ML Internship. The task focuses on implementing the **K-Nearest Neighbors (KNN)** algorithm using the **Iris dataset** from Kaggle to classify different species of flowers. The goal is to understand instance-based learning, how to choose the value of K, and visualize decision boundaries.

---

## Objective

Understand and implement KNN for classification problems

---

## Files Included

| File Name      | Description                                                             |
|----------------|-------------------------------------------------------------------------|
| `Iris.csv`     | Dataset used for classification                                         |
| `knn.ipynb`    | Jupyter Notebook containing full implementation and visualization       |
| `screenshots/` | Folder containing plots                                                 |
| `README.md`    | Project documentation                                                   |

---

## What I Did

**1. Dataset Selection and Loading**

I began by selecting the Iris dataset from Kaggle for this classification task. It's a simple, well-known dataset with 150 samples spread across three flower species: Iris-setosa, Iris-versicolor, and Iris-virginica. Each sample includes four features: sepal length, sepal width, petal length, and petal width.

I loaded the dataset using pandas and performed some initial checks using functions like .head(), .info(), and .describe() to get an overview of the data. I also verified there were no missing values and that the dataset was balanced across the three classes.

**2. Data Preprocessing**

Before training the model, I extracted the features (X) and labels (y) from the dataset. Since KNN is distance-based and sensitive to feature scales, I used StandardScaler from sklearn.preprocessing to normalize all features to have a mean of 0 and standard deviation of 1. This ensured that no feature would dominate the distance calculation due to its scale.

I then split the dataset into training and testing sets (80/20 split) using train_test_split() so that the model could be evaluated on unseen data.

**3. Model Training with K-Nearest Neighbors**
   
I used KNeighborsClassifier from sklearn.neighbors to build and train the KNN model. I initially set K = 3 and trained the model on the training data. After training, I used the model to make predictions on the test set.

**4. Evaluating Different K Values**

To find the most effective number of neighbors (K), I ran a loop to test K values from 1 to 10. For each K, I:

- Trained a new model
- Made predictions on the test set
- Calculated the accuracy

**5. Model Evaluation**

Once I had the best K value, I evaluated the model in more detail using:

- Accuracy score is used to measure the overall correctness of predictions.
- Confusion Matrix is used to visualize how many predictions were true positives, true negatives, false positives, and false negatives.

The confusion matrix provided an understanding of how well the KNN model was at classifying various flowers. The upper left cell of the confusion matrix told me that all 10 samples for the Iris-setosa species were predicted 'correctly'. The same pattern occurred for Iris-versicolor, where all 10 predicted were correct. On the Iris-virginica species, the KNN model predicted correctly on 8 of the samples and incorrectly predicted 2 samples as Iris-versicolor. 

After looking at those two samples, it prompted me to think about the feature space and perhaps virginica and versicolor share similar characteristics and create an overlapping area in the feature space which is a challenge for the model to separate. Regardless, the model performance accuracy was still relatively high, and this matrix reinforced that the classifier was, in general, making the right classification and doing so with very high confidence.

**6. Decision Boundary Visualization**

In order to see how KNN was actually separating the data, I trained one on just the first two features i.e., sepal length and sepal width. After training, I displayed the decision boundaries based on how the model would classify any new point in this two-dimensional feature space.

The first thing that struck me was how well Iris-setosa was separated from the other two species, it had its region all to itself with no overlap. The area of overlapping was more interesting, between Iris-versicolor and Iris-virginica. The green and blue areas didn't really have a border. Some points were right on the edge, thereby forcing the classifier to make tough decisions. I also found it somehow satisfying that visually, the plot corresponded to the confusion matrix results -- the same two misclassifications were apparent just the same. This type of visualization showed me understanding not only how good the model was doing, but also why it was confused sometimes, and where the limits in terms of decision-making are in a real geometric space.

## What I learned

This task helped me:

- Understand how KNN is an instance-based (lazy) learner that makes predictions based on closest training examples.
- See the impact of normalization on distance-based models.
- Learn how changing K balances the bias-variance tradeoff.
- Visualize how the algorithm classifies data in feature space using decision boundaries.
- Appreciate how even a simple algorithm like KNN can give high accuracy when properly tuned and visualized

---

## Libraries Used

- Python 3.12
- `pandas`, `numpy` — for data handling and preprocessing
- `matplotlib` — for visualization
- `scikit-learn` — for model training and evaluation

---
## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/task-6-knn-classification.git
   cd task-6-knn-classification

2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib scikit-learn

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook knn.ipynb

## Author

**Anmol Thakur**

GitHub: [anmolthakur74](https://github.com/anmolthakur74/)
