# Import necessary libraries
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import pickle

# Load the data into a numpy array

data = pd.read_csv("food.csv")

# Extract the input features
X = data[['veg_or_nonveg', 'Taste', 'Prep Time', 'Budget', 'Type']]

# Extract the output variable
y = data['Output']

# One-hot encode the categorical data
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X)
pickle.dump(enc, open('enc.pkl','wb'))
X = enc.fit_transform(X).toarray()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create the MLP model
mlp = MLPClassifier(hidden_layer_sizes=(20,), max_iter=100, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

# Train the model on the training data
mlp.fit(X_train, y_train)

# Print the accuracy of the model on the training data
print("Training set score: %f" % mlp.score(X_train, y_train))


# Define the input features you want to predict on
input_features = pd.DataFrame(
    {'veg_or_nonveg': ['Vegetarian'], 'Taste': ['Salty'], 'Prep Time': ['10Mins'], 'Budget': ['Low'],
     'Type': ['Snacks']})

# One-hot encode the input features
input_features = enc.transform(input_features).toarray()

# Predict the output labels for the given input features
predicted_output = mlp.predict(input_features)

# Print the predicted output
print("Predicted output:", predicted_output)

pickle.dump(mlp, open('model.pkl','wb'))
