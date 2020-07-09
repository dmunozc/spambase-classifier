"""Demonstration of a logistic regression and naive gaussian classifier.

It uses the spambase dataset.
"""


import numpy as np
import csv
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

# Avoids division by zero.
EPSILON = 0.001


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


def naive_gaussian_classifier(
    training_features, test_features, training_classes, probability_class,
):
    """Based on naive gaussian probability, return predictions."""
    predicted_class = []
    # Get gaussian probability statistics for each class.
    std_spam, mean_spam = gaussian_stats_by_class(
        training_features, training_classes, 1
    )
    std_not_spam, mean_not_spam = gaussian_stats_by_class(
        training_features, training_classes, 0
    )
    # For each feature calculate its probability based on the gaussian
    # formulas.
    for i in range(len(test_features)):
        probability_spam = math.log10(probability_class[0]) + np.sum(
            np.log10(
                (1 / (math.sqrt(2 * math.pi) * std_spam))
                * np.exp(
                    -(
                        np.power(test_features[i] - mean_spam, 2)
                        / (2 * np.power(std_spam, 2))
                    )
                )
            )
        )
        probability_not_spam = math.log10(probability_class[0]) + np.sum(
            np.log10(
                (1 / (math.sqrt(2 * math.pi) * std_not_spam))
                * np.exp(
                    -(
                        np.power(test_features[i] - mean_not_spam, 2)
                        / (2 * np.power(std_not_spam, 2))
                    )
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


def main():
    features = []
    class_types = []
    data_reader = csv.reader(open("spambase.csv"), delimiter=",")
    # Append all features and classes from csv file.
    for row in data_reader:
        features.append([float(i) for i in row[0:57]])
        class_types.append(int(row[-1]))

    features = np.array(features)
    class_types = np.array(class_types)
    # use Scikit's built in data splitter.
    (
        training_features,
        test_features,
        training_classes,
        test_classes,
    ) = train_test_split(
        features, class_types, test_size=0.5, random_state=None
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

    print("Logistic Regression Classifier Result")
    predicted_class = logistic_regresssion(
        training_features, test_features, training_classes
    )
    print("Confusion Matrix")
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


if __name__ == "__main__":
    main()
