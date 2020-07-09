"""Demonstration of a support vector machine classifier.

It uses the spambase dataset. Using the top 2 features it achieves an
accuracy of ~75%. Using all features it reaches ~92%.
"""

from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def features_training(
    training_features,
    test_features,
    training_classes,
    test_classes,
    feature_selection,
):
    """Perform SVM linear classification. Return accuracies.

    Given the features in order of preference, perform SVM training and
    get accuracy.
    It starts with the top 2 features, then top 3, etc, until using all
    features.
    """
    accuracies = []
    # Get the first training and test feature data based on the first index.
    training_data = np.array([training_features[:, feature_selection[0]]])
    test_data = np.array([test_features[:, feature_selection[0]]])
    # For all features, perform classification.
    for i in range(1, len(training_features[0])):
        # Append next features.
        training_data = np.append(
            training_data,
            np.array([training_features[:, feature_selection[i]]]),
            axis=0,
        )
        test_data = np.append(
            test_data,
            np.array([test_features[:, feature_selection[i]]]),
            axis=0,
        )

        # Transpose these data so features are in columns instead of rows.
        training_data_t = training_data.T
        test_data_t = test_data.T
        # Get stdev and mean for test data. Used for scaling.
        std_train = training_data_t.std(axis=0)
        mean_train = training_data_t.mean(axis=0)
        # Use scikit's preprocessing module to scale training data.
        training_data_sc = preprocessing.scale(training_data_t)
        # Scale test data based on mean and stdev of training data
        test_data_sc = test_data_t.copy()
        for i in range(len(test_data_sc)):
            test_data_sc[i] = (test_data_sc[i] - mean_train) / std_train
        # Perform classification.
        clf = svm.SVC(kernel="linear")
        clf.fit(training_data_sc, training_classes)
        prediction = clf.predict(test_data_sc)
        accuracies.append(accuracy_score(test_classes, prediction))
        # print accuracy of the training of the selected features
    return accuracies


def random_feature_selection(
    training_features, test_features, training_classes, test_classes,
):
    """Performs SVM classifaction using random features."""
    random_features = np.arange(57)
    # Randomize indices to access when training.
    np.random.shuffle(random_features)
    # Perform features training based on these random features.
    accuracies = features_training(
        training_features,
        test_features,
        training_classes,
        test_classes,
        random_features,
    )
    features = 2
    for acc in accuracies:
        print(
            "Using ",
            features,
            "random features, SVM achieves an accuracy " "of",
            acc,
        )
        features += 1


def weight_feature_selection(
    training_features, test_features, training_classes, test_classes, weights,
):
    """Performs SVM classifaction using best features."""
    sorted_features = np.argsort(weights)[::-1]
    # Perform features training based on these sorted indices
    accuracies = features_training(
        training_features,
        test_features,
        training_classes,
        test_classes,
        sorted_features,
    )
    features = 2
    for acc in accuracies:
        print(
            "Using top",
            features,
            "features, SVM achieves an accuracy " "of",
            acc,
        )
        features += 1


def svm_train(
    training_features, test_features, training_classes, test_classes,
):
    """Gets weights from SVM training."""
    # Get the mean and std from the training data.
    std_train = training_features.std(axis=0)
    mean_train = training_features.mean(axis=0)
    # Scale the test data using the mean and std from training data.
    test_features_sc = test_features.copy()
    for i in range(len(test_features_sc)):
        test_features_sc[i] = (test_features_sc[i] - mean_train) / std_train
    # Use Scikit's preprocessing module to scale training data.
    X = preprocessing.scale(training_features)
    y = training_classes
    # Use a linear classifier.
    clf = svm.SVC(kernel="linear")
    clf.fit(X, y)
    # Get the accuracy, precision, and recall.
    prediction = clf.predict(test_features_sc)
    accuracy = accuracy_score(test_classes, prediction)
    precision = precision_score(test_classes, prediction)
    recall = recall_score(test_classes, prediction)
    y_score = clf.decision_function(test_features_sc)

    # Plot ROC curve.
    fpr, tpr, thresholds = roc_curve(test_classes, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    plt.plot(fpr, tpr, color="red", label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([-0.01, 1.05])
    plt.ylim([-0.01, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")

    textstr = "\n".join(
        (
            r"$Accuracy=%.2f$" % (accuracy,),
            r"$Precision=%.2f$" % (precision,),
            r"$Recall=%.2f$" % (recall,),
        )
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

    # Place a text box with accuracy, precision, and recall.
    ax.text(
        0.8, 0.3, textstr, fontsize=14, verticalalignment="top", bbox=props
    )
    plt.draw()
    # return weight vector
    return clf.coef_


def main():
    features = []
    class_type = []
    data_reader = csv.reader(open("spambase.csv"), delimiter=",")
    # Get all features and classes from csv file.
    for row in data_reader:
        features.append([float(i) for i in row[0:57]])
        class_type.append(int(row[-1]))

    features = np.asarray(features)
    class_type = np.asarray(class_type)

    # Use Scikit's built in data splitter.
    (
        training_features,
        test_features,
        training_classes,
        test_classes,
    ) = train_test_split(
        features, class_type, test_size=0.5, random_state=None
    )

    # Get weights from SVM training.
    weights = svm_train(
        training_features, test_features, training_classes, test_classes,
    )[0]
    print("Weight based Feature selection")
    # use this weight vector to perform feature selection based on weight
    weight_feature_selection(
        training_features,
        test_features,
        training_classes,
        test_classes,
        weights,
    )
    print("Random Feature selection")
    # perform random feature selection
    random_feature_selection(
        training_features, test_features, training_classes, test_classes,
    )
    # plot ROC curve from experiment one
    plt.show()


if __name__ == "__main__":
    main()
