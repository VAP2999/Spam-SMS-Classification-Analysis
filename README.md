# Spam-SMS-Classification-Analysis


This repository contains the implementation of a data analysis and machine learning project focused on classifying SMS messages into **spam** and **ham** (non-spam). The project involves exploratory data analysis (EDA), feature engineering, and the application of various machine learning models to predict the category of a message.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Key Insights & EDA](#key-insights--eda)
- [Machine Learning Models](#machine-learning-models)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

---

## Project Overview

The goal of this project is to analyze SMS data to classify messages as either **spam** or **ham**. The project includes:
- **Exploratory Data Analysis (EDA)**: Visualization and examination of text features like character count, word count, sentence count, and their relationships to the message category.
- **Data Preprocessing**: Cleaning and tokenizing the text data, handling missing values, and feature engineering.
- **Machine Learning Models**: Building, training, and evaluating various classification models to predict whether an SMS message is spam or not.

---

## Dataset

The dataset used for this analysis is a collection of SMS messages labeled as either **spam** or **ham**. The dataset contains two primary columns:
- **category**: The label indicating whether the message is **spam** or **ham**.
- **text**: The content of the SMS message.

The dataset also includes additional preprocessing columns for character count, word count, and sentence count for each message.

---

## Key Insights & EDA

Several key insights were derived from the exploratory data analysis:
- **Character Count Analysis**: Spam messages tend to have a larger number of characters compared to ham messages.
- **Word Count**: Spam messages generally contain more words than ham messages.
- **Sentence Count**: Spam messages have an average of more sentences than ham messages.
- **Common Words in Spam**: Words like "free" appear more frequently in spam messages.

Visualizations include:
- **Pie charts** showing the distribution of spam and ham messages.
- **Box plots** to visualize the distribution of word, character, and sentence counts across both categories.
- **Correlation heatmap** to show relationships between text features.
- **Histograms** for comparing the distributions of text lengths (characters, words, and sentences) in spam vs ham messages.

---

## Machine Learning Models

Various machine learning models were implemented to classify SMS messages as spam or ham, including:
1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Decision Tree Classifier**
4. **Random Forest Classifier**
5. **Naive Bayes Classifier**
6. **Support Vector Machine (SVM)**

Each model was trained and evaluated on the features extracted from the SMS messages, with a focus on performance metrics such as **precision**, **accuracy**, and **confusion matrix**.

---

## Model Performance

The performance of the models was evaluated on the following metrics:
- **Precision**: The proportion of positive identifications (correctly predicted spam messages) out of all positive predictions.
- **Accuracy**: The proportion of correct predictions (both spam and ham) out of all predictions.
- **Confusion Matrix**: A detailed view of true positives, false positives, true negatives, and false negatives.

---

## Installation

To run this project on your local machine, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/spam-sms-classification.git
    ```

2. Navigate into the project directory:
    ```bash
    cd spam-sms-classification
    ```


4. Download the dataset `spam.csv` and place it in the `data` directory.

5. Run the project in Jupyter Notebook or Python:
    ```bash
    jupyter notebook
    ```

---

## Usage

1. Open the Jupyter Notebook file `spamfinal.ipynb`.
2. Execute the cells to load the data, clean it, perform exploratory data analysis (EDA), and train the machine learning models.
3. Analyze the visualizations and model performance.
4. Modify or extend the code to experiment with different features or models.

---

## Contributing

Contributions are welcome! If you want to contribute to the project, please follow these steps:
1. Fork the repository.
2. Create a new feature branch (`git checkout -b feature-name`).
3. Make your changes and commit them (`git commit -m "Description of changes"`).
4. Push to the branch (`git push origin feature-name`).
5. Create a pull request.

---




## Acknowledgments

- Libraries used: Pandas, NumPy, Matplotlib, Seaborn, Plotly, NLTK, Scikit-learn
- Tools: Jupyter Notebook, Python

