# Airbnb Machine Learning Analysis

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Data Description](#data-description)  
3. [Notebook Structure](#notebook-structure)  
4. [Installation & Setup](#installation--setup)  
5. [Usage](#usage)  
6. [Dependencies](#dependencies)  
7. [Results & Outputs](#results--outputs)  
8. [Author](#author)  

---

## Project Overview

This repository contains a Jupyter Notebook (`ML.ipynb`) that walks through a complete machine learning pipeline on an Airbnb listings dataset. The goals of the project are:

- **Data Ingestion & Merging**  
  Import multiple Airbnb‐related CSV files, concatenate them into a single DataFrame, and prepare for analysis.

- **Exploratory Data Analysis (EDA) & Visualization**  
  Explore summary statistics, visualize distributions of key features, and identify patterns or correlations.

- **Data Cleaning & Preprocessing**  
  Handle missing values, detect and treat outliers, perform feature selection, one‐hot encode categorical variables, and scale numeric features.

- **Dimensionality Reduction (PCA)**  
  Use Principal Component Analysis (PCA) to reduce feature dimensionality and visualize explained variance.

- **Unsupervised Learning (Clustering)**  
  Apply a clustering algorithm (e.g., K-Means) on the PCA‐transformed data, generate cluster summaries, and interpret group characteristics.

- **Supervised Learning (Regression Modeling)**  
  Split the data into training and testing sets, train multiple regression models (Linear Regression, Ridge, Lasso) to predict a target variable (e.g., listing price), evaluate performance (MSE, MAE, R²), and compare results.

By following this notebook step-by-step, you will learn how to take raw Airbnb data from ingestion all the way through model evaluation.

---

## Data Description

All raw data files (CSVs) are assumed to live in a folder called `Dataset/`. The notebook expects you to structure your files like this:

