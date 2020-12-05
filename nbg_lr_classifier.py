"""Demonstration of a logistic regression and naive gaussian classifier.

It uses the spambase dataset.
"""


import numpy as np
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

# Avoids division by zero.
EPSILON = 0.001


def confusion_matrix_metrics(test_classes, predicted_class):
    """
    Prints a confusion matrix and its metrics for 2 classes.
    """
    cm = confusion_matrix(test_classes, predicted_class)
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print(
        "Accuracy",
        (tp + tn) / (tn + tp + fp + fn),
        "precision",
        tp / (tp + fp),
        "recall",
        tp / (tp + fn),
    )

def logistic_regresssion(training_features, test_features, training_classes):
    """Wrapper function for Scikit's Logistic Regression"""
    model = LogisticRegression(solver="liblinear", multi_class="ovr")
    model.fit(training_features, training_classes)

    return model.predict(test_features)


def gaussian_stats_by_class(featuresList, classList, class_types):
    """Returns stdev and mean of a given class only."""
    # Get locations of features based on the class_types.
    indices_class = np.nonzero(classList != class_types)[0]
    # Remove the non desired classes from features.
    result = np.delete(featuresList, indices_class, axis=0)

    return result.std(axis=0) + EPSILON, result.mean(axis=0)


def naive_gaussian_classifier(training_features, test_features, training_classes, probability_class):
    """Based on naive gaussian probability, return predictions."""
    predicted_class = []
    # Get gaussian probability statistics for each class.
    std_spam, mean_spam = gaussian_stats_by_class(training_features, training_classes, 1)
    std_not_spam, mean_not_spam = gaussian_stats_by_class(training_features, training_classes, 0)

    # For each feature calculate its probability based on the gaussian formulas.
    for i in range(len(test_features)):
        probability_spam = np.log10(probability_class[0]) + np.sum(
            np.log10(
                (1 / (np.sqrt(2 * np.pi) * std_spam))
                * np.exp(
                    -(np.power(test_features[i] - mean_spam, 2) / (2 * np.power(std_spam, 2)))
                )
            )
        )
        probability_not_spam = np.log10(probability_class[0]) + np.sum(
            np.log10(
                (1 / (np.sqrt(2 * np.pi) * std_not_spam))
                * np.exp(
                    -(np.power(test_features[i] - mean_not_spam, 2) / (2 * np.power(std_not_spam, 2)))
                )
            )
        )
        # Choose the result based on which probability is greater.
        pred = 1 if probability_spam > probability_not_spam else 0
        predicted_class.append(pred)

    return np.array(predicted_class)


def prior_probability(classes, class_types):
    """Computes prior probabilities."""
    class_list = classes == class_types
    return np.count_nonzero(class_list) / len(classes)


def main(data_file):
    df = pd.read_csv(data_file, index_col=None, header=None)
    features = df.iloc[:, :57].values
    class_type = df.iloc[:, -1].values
    # use Scikit's built in data splitter.
    training_features, test_features, training_classes, test_classes, = train_test_split(
        features, class_type, test_size=0.5, random_state=2814
    )

    training_probability_spam = prior_probability(training_classes, 1)
    training_probability_not_spam = prior_probability(training_classes, 0)
    print("Naive Gaussian Classifier Result")
    predicted_class = naive_gaussian_classifier(
        training_features,
        test_features,
        training_classes,
        [training_probability_not_spam, training_probability_spam],
    )
    print("Confusion Matrix")
    confusion_matrix_metrics(test_classes, predicted_class)

    print("Logistic Regression Classifier Result")
    predicted_class = logistic_regresssion(
        training_features, test_features, training_classes
    )
    print("Confusion Matrix")
    confusion_matrix_metrics(test_classes, predicted_class)


if __name__ == "__main__":
    np.random.seed(2814)

    parser = argparse.ArgumentParser()
    parser.add_argument("-data", dest="data", help="spambase dataset", default="spambase.csv")
    args = parser.parse_args()
    main(args.data)
