# EDA_Project

# 🕋 Hajj & Umrah Crowd Management System

An interactive data-driven tool designed to enhance pilgrim safety through real-time crowd density analysis and technical recommendation logic.

## 📌 Project Overview
Crowd management during Hajj and Umrah is a high-stakes task where safety is the absolute priority. This project analyzes a dataset of 10,000 pilgrim records to identify patterns in crowd density, movement, and health conditions across various holy sites. 

Following a detailed Exploratory Data Analysis (EDA), the project implements a **Rule-Based Recommendation System (RS)**. This architecture was chosen to ensure maximum transparency and deterministic reliability, providing consistent safety guidance based on real-time data inputs.

## ⚖️ Technical Decision: Rule-Based Logic vs. Classification model
Based on the Exploratory Data Analysis (EDA), a rule-based Recommendation System approach was selected for the following reasons:
-  **Data Variability:** The Exploratory Data Analysis (EDA) revealed that the features did not have a strong mathematical correlation with the target outcomes (low signal-to-noise ratio). In such cases, a machine learning model would produce "confused" or inconsistent predictions. A deterministic rule-based system was chosen because it guarantees a specific, safe response for every input, ensuring that the system never provides a random or dangerous recommendation.
- **AI Ethics & Safety:** In high-density environments like the Holy Mosque, deploying a model without absolute certainty in its mapping is a safety risk. 
- **Interpretability:** A rule-based system provides clear, "if-then" logic that users and staff can immediately understand and trust.
- **Reliability:** By using predefined safety thresholds, the system avoids the unpredictability of classification errors, ensuring pilgrim well-being is always protected.

## 🚀 Features
- **Comprehensive EDA:** Statistical analysis of factors like temperature, movement speed, and activity types.
- **Interactive Interface:** A user-friendly dashboard built with **Streamlit** for real-time monitoring.
- **Actionable Recommendations:**
  - 🔴 **High Density:** Guidance to avoid the area and follow staff directions.
  - 🟡 **Medium Density:** Instructions to move slowly and maintain distance.
  - 🟢 **Low Density:** Identification of optimal times for rituals.

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn
- **Web Framework:** Streamlit
- **Report Writing:** LaTeX (Overleaf)

## 📊 Dataset Information
- **Source:** Kaggle
- **Volume:** 10,000 entries | 30 features
- **Key Attributes:** `Crowd_Density`, `Movement_Speed`, `Weather_Conditions`, `Health_Condition`.

