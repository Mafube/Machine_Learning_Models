# About this Project
This project aims to develop a robust machine-learning model for disease classification based on symptoms. By leveraging multiple machine learning algorithms, the system will analyze the given symptoms and predict the most likely disease. The goal is to improve the accuracy and efficiency of early disease detection, which can assist healthcare professionals in decision-making. The dataset used for training and evaluation consists of various symptoms and their corresponding diseases. The model is evaluated based on performance metrics to ensure reliability.

# Language Used
- Python: The core programming language used for data preprocessing, model training, and evaluation.

- Pandas: Used for data manipulation, cleaning, and structuring the dataset before feeding it into machine learning models.

# Machine Learning Algorithms Used
- **K-Fold Cross-Validation:**

A resampling technique used to assess the model’s generalizability and prevent overfitting.

It splits the dataset into multiple subsets, trains the model on different combinations, and evaluates performance on the remaining subset.

- **Support Vector Classifier (SVC):**

A supervised learning model used for classification tasks.

Works by finding the optimal hyperplane that maximizes the margin between different disease classes.

Useful for high-dimensional datasets and can be effective in handling complex decision boundaries.

- **Gaussian Naive Bayes Classifier:**

A probabilistic classifier based on Bayes' Theorem.

Assumes features are normally distributed and calculates the likelihood of a disease given a set of symptoms.

Efficient for classification problems with a small dataset and performs well when the independence assumption holds.

- **Random Forest Classifier:**

An ensemble learning method that constructs multiple decision trees during training.

Uses majority voting from different trees to improve classification accuracy and reduce overfitting.

Effective in handling large datasets with multiple features and can provide feature importance rankings.

This project aims to compare the performance of these algorithms and develop a final prediction model that balances accuracy, efficiency, and real-world applicability in disease classification.

