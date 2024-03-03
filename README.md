# Credit-Card-Detection-Prediction

This Python code performs credit card fraud detection using logistic regression, with an emphasis on handling imbalanced data using undersampling and oversampling techniques. Let's break down the code step by step:

Importing Libraries: The code begins by importing necessary libraries including NumPy, Pandas, Seaborn, Matplotlib, and scikit-learn modules for data manipulation, visualization, model training, and evaluation.

Loading the Dataset: The code loads a CSV file containing credit card transaction data into a Pandas DataFrame (main_df), and then creates a copy of it named df.

Data Exploration:

Information about the dataset is printed using df.info() to understand its structure and data types.
Null values in the dataset are checked using df.isnull().sum().
The distribution of the target variable "Class" (0 for normal transactions, 1 for fraud transactions) is printed using df['Class'].value_counts().
Undersampling:

Undersampling is performed to balance the classes by randomly selecting a subset of normal transactions (legit_sample) to match the number of fraud transactions.
The balanced dataset is created by concatenating legit_sample and fraud transactions into new_df.
Splitting Data:

Features (X) and target (Y) are separated.
The data is split into training and testing sets using train_test_split.
Model Training and Evaluation (Before SMOTE):

Logistic Regression model is trained on the original dataset (X_train, y_train).
Predictions are made on both training and testing sets.
Accuracy scores and confusion matrix are computed.
ROC curve is plotted.
Oversampling (SMOTE):

SMOTE (Synthetic Minority Over-sampling Technique) is applied to oversample the minority class (fraud transactions) in the original dataset (df) to create a balanced dataset (X_oversampled, y_oversampled).
Model Training and Evaluation (After SMOTE):

Logistic Regression model is trained on the oversampled dataset.
Predictions are made on both training and testing sets.
Accuracy scores and confusion matrix are computed.
ROC curve is plotted.
Summary:

The code concludes by printing the original and transformed fraud class distributions and presenting the classification report.
Overall, this code demonstrates how to address class imbalance in a classification problem, specifically for credit card fraud detection, using logistic regression and techniques like undersampling and oversampling.



# Exploring Credit Card Fraud Detection: Data Analysis, Model Training, and Outlier Detection

Importing Modules: The code begins by importing necessary Python libraries such as NumPy, Pandas, Matplotlib, Seaborn, and GridSpec for data manipulation, visualization, and plotting.

Reading the Dataset: The next step involves reading a CSV file named "creditcard.csv" into a Pandas DataFrame called dataset. This dataset likely contains information about credit card transactions, including features like transaction amount, time, and possibly other anonymized features.

Data Visualization: Various visualizations are created to understand the data better. This includes:

Checking the relative proportion of fraudulent and valid transactions.
Visualizing the distribution of transaction amounts and times.
Plotting the distributions of individual features, comparing fraudulent and valid transactions.
Data Preparation:

Checking for null values in the dataset.
Scaling the "Time" and "Amount" features using RobustScaler.
Separating the response variable ("Class", indicating fraudulent or valid transaction) from the features.
Splitting the data into training and testing sets using train_test_split.
Model Training:

Creating a cross-validation framework using StratifiedKFold.
Importing classifiers from Scikit-learn and the imbalanced-learn module for handling class imbalance.
Training a Random Forest Classifier on the training data.
Evaluating the model's performance using metrics such as accuracy, precision, recall, and F1 score.
ROC Curve Analysis:

Calculating and plotting Receiver Operating Characteristic (ROC) curves to assess model performance.
ROC curves help visualize the trade-off between true positive rate (sensitivity) and false positive rate (1 - specificity) for different threshold values.
Outlier Detection and Removal:

Visualizing the distribution of selected features using boxplots.
Calculating the Interquartile Range (IQR) to identify outliers.
Removing outliers from the dataset based on the IQR method.
This code essentially performs exploratory data analysis, model training, evaluation, and outlier detection in the context of credit card fraud detection. It utilizes various Python libraries and techniques to gain insights from the data and build a predictive model to detect fraudulent transactions.
