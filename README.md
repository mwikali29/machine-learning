 AI for Diabetes Risk Prediction (SDG 3)
Project Overview
This project demonstrates the application of a supervised machine learning model to predict the risk of diabetes based on diagnostic measurements. Aligned with UN Sustainable Development Goal 3: Good Health and Well-being, the solution aims to facilitate early detection and proactive management of diabetes, thereby contributing to improved health outcomes and a more sustainable healthcare system.

 UN SDG & Problem Addressed
UN SDG 3: Good Health and Well-being

Specific Problem: Early detection and prevention of Non-Communicable Diseases (NCDs), particularly diabetes. Early identification of individuals at risk can enable timely interventions, lifestyle changes, and better disease management, reducing the burden on healthcare systems and improving quality of life.

Machine Learning Approach
Type: Supervised Learning (Classification)

Model: Logistic Regression
Why? It's a robust, interpretable, and efficient algorithm suitable for binary classification tasks (diabetic vs. non-diabetic), providing probabilities which are useful in a medical context.

 Dataset
Source: Pima Indians Diabetes Dataset (publicly available from sources like Kaggle and UCI Machine Learning Repository).
Description: The dataset comprises diagnostic measurements from a specific population (Pima Indian women) and includes several medical predictor variables and one target variable (Outcome).
Features (X): Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.
Target (y): Outcome (0 for non-diabetic, 1 for diabetic).

How to Run the Code
Prerequisites: Ensure you have Python installed, along with the following libraries:
pandas
numpy
scikit-learn
matplotlib 
seaborn
You can install them via pip:
pip install pandas numpy scikit-learn matplotlib seaborn
Download the Code: Copy the Python code provided in the "Diabetes Risk Prediction Model" immersive.
Save the Code: Save it as a Python file (e.g., diabetes_prediction_model.py) or paste it directly into a Jupyter Notebook cell.
Run:
From Terminal: Navigate to the directory where you saved the file and run:
           python diabetes_prediction_model.py
In Jupyter Notebook/Colab: Paste the code into a cell and run the cell.
The script will:
Load the dataset (from a URL or use dummy data if loading fails).
Perform basic data preprocessing (handling zeros, splitting, scaling).
Train the Logistic Regression model.
Evaluate the model and print metrics (accuracy, classification report, confusion matrix).
Display a confusion matrix plot.
Show feature coefficients for interpretability.

Key metrics you'll see in the output:
Accuracy: Overall correctness of predictions.
Precision, Recall, F1-score: Measures of the model's ability to correctly identify positive cases (diabetic) and avoid false alarms.
Confusion Matrix: Shows the count of True Positives, True Negatives, False Positives, and False Negatives.
Feature Coefficients: Indicates the relative importance and direction of influence for each input feature on the prediction.

Ethical Considerations
Data Bias: The primary ethical concern is the bias inherent in the dataset. The Pima Indians Diabetes Dataset is specific to Pima Indian women. A model trained solely on this data may not generalize well to other populations (different ethnicities, genders, ages, or socio-economic backgrounds). Deploying such a model broadly without addressing this bias could lead to inaccurate predictions and exacerbate existing health disparities.

Fairness: To promote fairness, future work must involve training on diverse and representative datasets that reflect the target population. Continuous monitoring and evaluation of model performance across different demographic subgroups are essential.

Transparency & Accountability: While Logistic Regression is relatively interpretable , complex AI models in healthcare require clear explanations for their predictions. Human oversight remains crucial; AI should be an assistive tool for medical professionals, not a replacement.

Privacy: Handling sensitive health data requires strict adherence to privacy regulations and robust data security measures.

Contribution to Sustainability (SDG 3)
By enabling earlier and more systematic identification of individuals at high risk of diabetes, this AI solution supports preventative healthcare strategies. This proactive approach can lead to:

Improved Patient Outcomes: Early lifestyle interventions and medical management can prevent or delay the onset of type 2 diabetes.

Reduced Healthcare Burden: Less severe disease progression means fewer hospitalizations, complications, and associated costs.
![image](https://github.com/user-attachments/assets/08c69346-900f-406b-ad60-9ff21278acc0)
![image](https://github.com/user-attachments/assets/87486226-e748-483a-867c-68bfdfb56b1c)
![image](https://github.com/user-attachments/assets/c0f4de32-437e-4e39-afb7-931b63d620ae)
![image](https://github.com/user-attachments/assets/5b801e32-ccc7-430b-86a5-c71384b18b9e)


Efficient Resource Allocation: Healthcare resources can be targeted more effectively towards individuals who need intervention most, contributing to the long-term sustainability and resilience of health systems.
