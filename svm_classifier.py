from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc
import numpy as np
import csv
import matplotlib.pyplot as plt
import random
import sys
from sklearn.model_selection import train_test_split


def featuresTraining(trainingFeatureData, testFeatureData, trainingClassType, 
                     testClassType, selectionIndices):
  #get the first training and test feature data based on the first index
  trainingData = np.array([trainingFeatureData[:,selectionIndices[0]]])
  testData = np.array([testFeatureData[:,selectionIndices[0]]])
  
  #iterate over all features
  for m in range(1, 57, 1):

    #append the next feature
    trainingData = np.append(trainingData, 
                             np.array([trainingFeatureData[:,selectionIndices[m]]]), axis=0)
    testData = np.append(testData, 
                         np.array([testFeatureData[:,selectionIndices[m]]]), axis=0)

    #transpose these data so features are in columns instead of rows
    trainingDataT = trainingData.T
    testDataT = testData.T
    #get stdev and mean for scaling test data
    stdDevTraining = trainingDataT.std(axis=0)
    meanTraining = trainingDataT.mean(axis=0)
    
    #Use scikit's preprocessing module to scale training data
    trainingDataS = preprocessing.scale(trainingDataT)
    #Scale test data based on mean and stdev of training data
    testDataS = np.array(testDataT, copy=True)
    for i in range(len(testDataS)):
      testDataS[i] = (testDataS[i] - meanTraining)/stdDevTraining
      
    #use a linear kernel
    clf = svm.SVC(kernel='linear')
    clf.fit(trainingDataS, trainingClassType) 
    prediction = clf.predict(testDataS)
    accuracy = accuracy_score(testClassType, prediction)
    #print accuracy of the training of the selected features
    print(accuracy)

def randomFeatureSelection(trainingFeatureData, testFeatureData, trainingClassType, testClassType):
  randomIndices = [x for x in range(57)]
  #randomize indices to access when training
  for _ in range(7):
    random.shuffle(randomIndices)
  #perform features training based on these random indices
  featuresTraining(trainingFeatureData, testFeatureData, trainingClassType, testClassType, np.array(randomIndices))  
 
 
 
def weightFeatureSelection(trainingFeatureData, testFeatureData, trainingClassType, testClassType, weightVector):
  #get the sorted indices of the weights vector
  sortedWeightsIndices = np.argsort(weightVector)[::-1]
  #perform features training based on these sorted indices
  featuresTraining(trainingFeatureData, testFeatureData, trainingClassType, testClassType, sortedWeightsIndices)



def svm(trainingFeatureData, testFeatureData, trainingClassType, testClassType):
  #get the mean and std from the training data
  stdDevTraining = trainingFeatureData.std(axis=0)
  meanTraining = trainingFeatureData.mean(axis=0)
  #use Scikit's preprocessing module to scale training data
  trainingFeatureDataS = preprocessing.scale(trainingFeatureData)
  #scale the test data using the mean and std from training
  testFeatureDataS = np.array(testFeatureData, copy=True)
  for i in range(len(testFeatureDataS)):
    testFeatureDataS[i] = (testFeatureDataS[i] - meanTraining)/stdDevTraining
    
    
  X = trainingFeatureDataS
  y = trainingClassType
  #use a linear classifer
  clf = svm.SVC(kernel='linear')
  clf.fit(X, y)  
  #get the accuracy, precision, and recall
  prediction = clf.predict(testFeatureDataS)
  accuracy = accuracy_score(testClassType, prediction)
  precision = precision_score(testClassType, prediction)
  recall = recall_score(testClassType, prediction)
  y_score = clf.decision_function(testFeatureDataS)
  
  #print ROC curve
  fpr, tpr, thresholds = roc_curve(testClassType, y_score)
  roc_auc = auc(fpr, tpr)
  fig, ax = plt.subplots()
  plt.plot(fpr, tpr, color='red', label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
  plt.xlim([-0.01, 1.05])
  plt.ylim([-0.01, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC Curve Experiment 1')
  plt.legend(loc="lower right")
  
  textstr = '\n'.join((
    r'$Accuracy=%.2f$' % (accuracy, ),
    r'$Precision=%.2f$' % (precision, ),
    r'$Recall=%.2f$' % (recall, )))
  props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
  
  # place a text box with accuracy, precision, and recall
  ax.text(0.8, 0.3, textstr, fontsize=14, verticalalignment='top', bbox=props)
  plt.draw()
  return clf.coef_ #return weight vector


def main():
  features = []
  classType = []
  dataFile = open(sys.argv[1])
  dataReader = csv.reader(dataFile, delimiter=',')
  #append all features and classes from csv file
  for row in dataReader:
    features.append([float(i) for i in row[0:57]])
    classType.append(int(row[-1]))
  
  features = np.asarray(features)
  classType = np.asarray(classType)
  
  #use Scikit's built in data splitter 
  trainingFeatureData, testFeatureData, trainingClassType, testClassType=train_test_split(features, classType, test_size=.5, random_state=None) 
  
  #get the weight vector using experiment's one training
  expOneWeightVector = svm(trainingFeatureData, testFeatureData, trainingClassType, testClassType)[0]
  print("Weight based Feature selection")
  #use this weight vector to perform feature selection based on weight
  weightFeatureSelection(trainingFeatureData, testFeatureData, trainingClassType, testClassType, expOneWeightVector)
  print("Random Feature selection")
  #perform random feature selection
  randomFeatureSelection(trainingFeatureData, testFeatureData, trainingClassType, testClassType)
  #plot ROC curve from experiment one
  plt.show()
  

if __name__ == '__main__':
  if len(sys.argv) > 1:
    main()
  else:
    print("Needs to provide spambase.csv as argument")