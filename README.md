# Neural Network Charity Analysis

## Overview of the Analysis
The purpose of this analysis was to use machine learning and neural networks to create a binary classifier that is capable of predicting whether applicants will be succesful if funded by Alphabet Soup.

### Resources
- Data Source: [charity_data.csv](https://github.com/dgornowicz/neural-network-charity-analysis/blob/main/charity_data.csv)
- Software: Python, Anaconda Navigator, Conda, Jupyter Notebook, Google Colab

## Results

### Data Processing
- The columns `EIN` and `NAME` are identification information and have been removed from the input data.
- The column `IS_SUCCESSFUL` contains binary data refering to weither or not the charity donation was used effectively. This variable is then considered as the target for our deep learning neural network.
- The following columns `APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT` are the features for our model.
Encoding of the categorical variables, spliting into training and testing datasets and standardization have been applied to the features.

### Compiling, Training, and Evaluating the Model
- This deep-learning neural network model is made of two hidden layers with 80 and 30 neurons respectively.
- The input data has 43 features and 25,724 samples.
- The output layer is made of a unique neuron as it is a binary classification.
- To speed up the training process, we are using the activation function `ReLU` for the hidden layers. As our output is a binary classification, `Sigmoid` is used on the output layer.
For the compilation, the optimizer is `adam` and the loss function is `binary_crossentropy`.
- The model accuracy is under 75%. This is not a satisfying performance to help predict the outcome of the charity donations.
- To increase the performance of the model, we applied bucketing to the feature `ASK_AMT` and organized the different values by intervals.
- We increased the number of neurons on one of the hidden layers, then we used a model with three hidden layers.
- We tried a different activation function (`tanh`) but none of these steps helped improve the model's performance.
- We finally tried synthesizing all of the optimizations into another model but this method also failed to improve the model's performance

## Summary
Model Accuracy and Loss:
- Original Attempt = 0.564 Loss, 0.727 Accuracy
- Adding more neurons, Second Attempt = 0.567 Loss, 0.729 Accuracy
- Adding more layers, Third Attempt = 0.571 Loss, 0.728 Accuracy
- Using different functions, Fourth Attempt = 0.562 Loss, 0.729 Accuracy
- Syntesize optimizations, Fifth Attempt = 0.573 Loss, 0.727 Accuracy

### Recommendation
Since we are in a binary classification situation, we could use a supervised machine learning model such as the Random Forest Classifier to combine a multitude of decision trees to generate a classified output and evaluate its performance against our deep learning model.
