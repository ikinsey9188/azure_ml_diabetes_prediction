# **Diabetes Prediction Project**

## **Project Overview**

This project aims to develop a robust machine learning model that can predict the likelihood of a patient being diabetic based on various health metrics. The project involves exploring multiple machine learning algorithms, comparing their performance, and selecting the best-performing model for potential deployment. The entire process is tracked and managed using MLflow, an open-source platform for managing the complete machine learning lifecycle, including experimentation, tracking, model management, and deployment. Azure Machine Learning Studio is leveraged for model deployment.

## **Objectives**

1. **Data Preparation:**
   - Load and preprocess the diabetes dataset, ensuring the data is clean and ready for model training.
   - Split the dataset into training and test sets to evaluate the model's performance.

2. **Model Training and Evaluation:**
   - Train multiple machine learning models, including Logistic Regression, Decision Tree, and Random Forest.
   - Utilize MLflow's autologging feature to automatically track the parameters, metrics, and artifacts for each model.
   - Manually log additional metrics (e.g., accuracy) and compare the performance of each model.

3. **Model Selection:**
   - Analyze the logged metrics to identify the best-performing model based on accuracy.
   - Register the best model in Azure ML for future use, ensuring it is ready for deployment.

4. **Model Deployment (Optional):**
   - Optionally deploy the best model as a web service in Azure, making it accessible for real-time predictions in external applications.

## **Process Overview**

1. **Environment Setup:**
   - Verify that the necessary libraries and SDKs are installed, including `azure-ai-ml`, `mlflow`, `scikit-learn`, and others required for model training and tracking.

2. **Data Loading and Preprocessing:**
   - Load the diabetes dataset from a CSV file.
   - Split the data into features (input variables) and labels (output variable) and further split it into training and test sets.

3. **Experiment Setup:**
   - Initialize an MLflow experiment to group all related model training runs.
   - Enable MLflow autologging to automatically track model parameters, metrics, and artifacts.

4. **Model Training:**
   - Train multiple machine learning models using different algorithms.
   - For each model, log the accuracy and other relevant metrics to MLflow.
   - Save the trained model as an artifact for future reference or deployment.

5. **Model Evaluation and Selection:**
   - Compare the performance of the different models based on the logged metrics.
   - Select the best model and register it in Azure ML for potential deployment.

6. **Model Registration and Deployment:**
   - Register the best-performing model in Azure ML to ensure it is easily accessible for deployment.
   - Optionally, deploy the model as a web service for real-time predictions.

7. **Project Documentation:**
   - Document the entire process, including data exploration, model selection criteria, and the final decision-making process.

## **How to Run the Project**

### **Prerequisites**
Before running the project, ensure you have the following installed:

- Python 3.8 or higher
- Jupyter Notebook
- Required Python libraries:
  - `azure-ai-ml`
  - `mlflow`
  - `scikit-learn`
  - `pandas`
  - `joblib`

You can install the necessary Python packages using the following command:

```bash
pip install azure-ai-ml mlflow scikit-learn pandas joblib
