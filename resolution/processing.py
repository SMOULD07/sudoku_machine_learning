import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def get_data(file):
    data = pd.read_csv(file,nrows=100000)

    quizzes = data["quizzes"]
    solutions = data["solutions"]

    labels=[]
    features=[]

    for i in quizzes:
        x =  np.array([int(j) for j in i]).reshape((9,9,1))
        features.append(x)

    features = np.array(features)
    features = (features/9) - 0.5


    for i in solutions:
        x = np.array([int(j) for j in i]).reshape((81,1)) - 1
        labels.append(x)
    
    labels = np.array(labels)

    del(quizzes)
    del(solutions)


    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test


