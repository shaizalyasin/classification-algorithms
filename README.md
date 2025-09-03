# Classification Algorithms: Decision Trees and Naïve Bayes  

This repository contains the implementation of two classic classification algorithms: **Decision Tree Induction** and **Naïve Bayes Classification**. The project focuses on building a functional framework for these algorithms to handle both **categorical** and **continuous** data.  

---

## 📊 Core Algorithms and Features  

### 🔹 Decision Tree Induction  
A Decision Tree classifier predicts outcomes by traversing a tree structure from root to leaf.  

**Key Features:**  
- **Tree Construction:** Build trees using decision nodes (`DecisionTreeInternalNode`) and prediction nodes (`DecisionTreeLeafNode`).  
- **Data Splitting:** Determine optimal split conditions based on attribute types and values.  
- **Prediction:** Traverse the tree to classify new inputs by following matching branches.  

---

### 🔹 Naïve Bayes Classification  
A probabilistic classification algorithm based on **Bayes’ theorem**, capable of handling categorical and continuous data.  

**Key Features:**  
- **Prior Probabilities:** Calculate the probability of each class in the dataset.  
- **Likelihoods:** Compute conditional probabilities of attribute values given a class.  
  - Supports categorical attributes.  
  - Uses Gaussian distribution for continuous attributes.  
- **Classification:** Combine priors and likelihoods to predict the most probable class for new data points.  

---

## 📌 Summary  
This project provides practical implementations of **Decision Trees** and **Naïve Bayes**, two foundational algorithms in **classification tasks**, useful for both educational purposes and as lightweight tools for small-scale data analysis.  
