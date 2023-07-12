# ML  Classification Models

Technology: Python (pandas, numpy, matplotlib, seaborn, sklearn libraries)
Dataset: https://www.kaggle.com/anjusukumaran4/code

Implemented ML Models : 

❄️ Linear Classifiers: 

Linear models create a linear decision boundary between classes. They are simple and computationally efficient. 

🔸 Logistic Regression

🔸 Support Vector Machines having kernel = ‘linear’

🔸 Single-layer Perceptron

🔸 Stochastic Gradient Descent (SGD) Classifier

❄️ Non-linear Classifiers:

Non-linear models create a non-linear decision boundary between classes. They can capture more complex relationships between the input features and the target variable.

🔸 K-Nearest Neighbours

🔸 Kernel SVM

🔸 Naive Bayes

🔸 Decision Tree Classification

🔸 Ensemble learning classifiers:  Random Forests, AdaBoost etc

🔸 Multi-layer Artificial Neural Networks

Steps involves :

1. Understanding the problem: Before getting started with classification, it is important to understand the problem you are trying to solve. What are the class labels you are trying to predict? What is the relationship between the input data and the class labels etc

2. Data preparation: Once you have a good understanding of the problem, the next step is to prepare your data. This includes collecting and preprocessing the data and splitting it into training, validation, and test sets. In this step, the data is cleaned, preprocessed, and transformed into a format that can be used by the classification algorithm.

3. Feature Extraction: The relevant features or attributes are extracted from the data that can be used to differentiate between the different classes.
Suppose our input X has 7 independent features, having only 5 features influencing the label or target values remaining 2 are negligibly or not correlated, then we will use only these 5 features only for the model training. 

4. Model Selection: There are many different models that can be used for classification, including logistic regression, decision trees, support vector machines (SVM), or neural networks. It is important to select a model that is appropriate for your problem, taking into account the size and complexity of your data, and the computational resources you have available.
   
5. Model Training: Once you have selected a model, the next step is to train it on your training data. This involves adjusting the parameters of the model to minimize the error between the predicted class labels and the actual class labels for the training data.
   
6. Model Evaluation: Evaluating the model: After training the model, it is important to evaluate its performance on a validation set. This will give you a good idea of how well the model is likely to perform on new, unseen data. Log Loss or Cross-Entropy Loss, Confusion Matrix,  Precision, Recall, and AUC-ROC curve are the quality metrics used for measuring the performance of the model.
   
7. Fine-tuning the model:  If the model’s performance is not satisfactory, you can fine-tune it by adjusting the parameters, or trying a different model.
    
8. Deploying the model: Finally, once we are satisfied with the performance of the model, we can deploy it to make predictions on new data.  it can be used for real world problem.

