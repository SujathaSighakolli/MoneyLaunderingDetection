# A Hybrid Modelling Approach for Detecting Money Laundering in Banking Sectors

## 📌 Project Overview

This project focuses on detecting money laundering activities in banking transactions using a hybrid combination of Machine Learning (ML) and Deep Learning (DL) algorithms. Traditional rule-based fraud detection systems often fail to accurately identify suspicious transactions. To overcome this limitation, this project introduces a hybrid intelligent system that analyzes transaction behavior and predicts whether a transaction is normal or suspicious.

The system uses multiple ML and DL algorithms along with advanced data preprocessing techniques to improve prediction accuracy and risk profiling.



## 🎯 Objectives

* Detect suspicious banking transactions using AI-based models
* Improve fraud detection accuracy compared to traditional rule-based systems
* Apply hybrid ML and DL techniques for better prediction
* Perform risk profiling based on user transaction behavior
* Handle imbalanced datasets using SMOTE



## 🛠 Technologies Used

* Python 3.7.2
* Django
* Machine Learning
* Deep Learning
* HTML, CSS, Bootstrap
* Kaggle AML Dataset



## 📚 Algorithms Used

The following algorithms are used and compared in this project:

* Support Vector Machine (SVM)
* Random Forest
* Decision Tree
* Naive Bayes
* Logistic Regression
* CNN2D



## ⚙️ Data Preprocessing Techniques

To improve model performance, the following preprocessing methods are applied:

* Missing value handling
* Feature shuffling
* Data normalization
* SMOTE for imbalance data handling
* Train-test splitting



## 📊 Performance Evaluation Metrics

Each algorithm is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score

Among all algorithms, **Random Forest** and **Logistic Regression** achieved the highest accuracy.



## 🧩 Project Modules

### 1. User Login

Users can log in using:

* Username: `admin`
* Password: `admin`



### 2. Load & Process Dataset

This module performs:

* Dataset loading
* Missing value replacement
* SMOTE balancing
* Data normalization
* Data shuffling
* Train-test split

Dataset split:

* 80% Training Data
* 20% Testing Data


### 3. Train Hybrid Algorithms

This module:

* Trains all ML and DL models
* Tests the models using test data
* Displays:

  * Accuracy
  * Precision
  * Recall
  * F1-Score
* Generates confusion matrix graphs
* Compares algorithms using graphical analysis



### 4. Predict Money Laundering

Users can enter transaction details such as:

* Transaction amount
* Location
* Other transaction parameters

The system predicts:

* Suspicious behavior ratio
* Risk profiling percentage
* Final transaction status:

  * Normal
  * Suspicious



# 📂 Dataset Information

Dataset Source: Kaggle AML Dataset

The dataset contains:

* Transaction records
* Transaction amount
* Location details
* User behavior patterns
* Fraud labels



# 🚀 Installation Steps

## Step 1: Install Python

Install:

* Python 3.7.2

---

## Step 2: Install Required Packages

Open terminal or command prompt and run:

```bash
pip install -r requirements.txt
```

---

## Step 3: Start the Server

Double-click the `run.bat` file
or run:

```bash
python manage.py runserver
```

---

## Step 4: Open the Application

Open browser and enter:

```text
http://127.0.0.1:8000/index.html
```

---

# 📸 System Workflow

## Home Page

* Start the application
* Navigate to user login

## User Login

* Login using admin credentials

## Dataset Processing

* Load AML dataset
* Apply preprocessing techniques
* Balance dataset using SMOTE

## Model Training

* Train all hybrid algorithms
* Display performance metrics
* Generate confusion matrix graphs

## Prediction Module

* Enter transaction details
* Predict suspicious or normal transaction
* Display risk profiling ratio



# 📈 Results

* Hybrid algorithms achieved better fraud detection performance
* Random Forest and Logistic Regression provided highest accuracy
* SMOTE improved class balancing and prediction quality
* The system successfully predicts suspicious transactions based on behavior analysis



# 🔮 Future Enhancements

* Real-time banking transaction monitoring
* Integration with cloud platforms
* Advanced deep learning models
* Explainable AI for fraud analysis
* Mobile application support



# 👩‍💻 Author

Developed as an academic final year project for detecting money laundering activities using Hybrid Machine Learning and Deep Learning techniques.



# 📄 License

This project is developed for educational and research purposes only.
