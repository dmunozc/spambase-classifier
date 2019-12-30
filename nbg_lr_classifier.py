import numpy as np
import csv
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import sys

EPSILON  = 0.0001


def logisticRegression(trainingFeatureData, testFeatureData, trainingClassType):
  model = LogisticRegression(solver='liblinear', multi_class='ovr')
  model.fit(trainingFeatureData, trainingClassType)
  
  return model.predict(testFeatureData)


#returns stdev and mean of the features based on the class given 
#and the features list. 
def gaussianStatsByClass(featuresList, classList, classType):
  #get locations of features based on the classType
  indicesOfClass = np.nonzero(classList != classType)[0]
  #remove the non desired classes from features
  result = np.delete(featuresList, indicesOfClass, axis=0)
  
  return result.std(axis=0) + EPSILON, result.mean(axis=0)

def naiveGaussianClassifier(trainingFeatureData, testFeatureData, trainingClassType, 
                            probabilityClassType):
  predictedClass= []
  #get gaussian probability statistics for each class
  stdSpam, meanSpam = gaussianStatsByClass(trainingFeatureData, trainingClassType, 1)
  stdNotSpam, meanNotSpam = gaussianStatsByClass(trainingFeatureData, trainingClassType, 0)
  #for each feature calculate its probability based on the gaussian formulas
  for i in range(len(testFeatureData)):
    probabilitySpam =math.log10(probabilityClassType[0]) + \
        np.sum(np.log10((1/(math.sqrt(2*math.pi)*stdSpam)) * \
                        np.exp(-(np.power(testFeatureData[i]-meanSpam,2)/(2*np.power(stdSpam,2))))))
    probabilityNotSpam = math.log10(probabilityClassType[0]) + \
        np.sum(np.log10((1/(math.sqrt(2*math.pi)*stdNotSpam)) * \
                        np.exp(-(np.power(testFeatureData[i]-meanNotSpam,2)/(2*np.power(stdNotSpam,2))))))
    #choose the result based on which probability is greater
    predClassType = 1 if probabilitySpam > probabilityNotSpam else 0
    predictedClass.append(predClassType)
    
  return np.asarray(predictedClass)
  
def computePriorProbability(classList, classType):
  #sets all elements in class list that are equal to classType to True, else is False
  classList = classList == classType
  #return instances of classType by class list length, which is the probability 
  return np.count_nonzero(classList)/len(classList)
    

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
  trainingFeatureData, testFeatureData,trainingClassType, testClassType = \
          train_test_split(features,classType, test_size=.5, random_state=None) 
  #compute prior probablities of the data
  trainingProbabilitySpam = computePriorProbability(trainingClassType, 1)
  trainingProbabilityNotSpam = computePriorProbability(trainingClassType, 0)
  
  print("Naive Gaussian Classifier Result")
  #get the predicted classes from the gaussian classifier
  predictedClass = naiveGaussianClassifier(trainingFeatureData, testFeatureData,
                                           trainingClassType, 
                                           [trainingProbabilityNotSpam,trainingProbabilitySpam])
  #create the confusion matrix based on the predicted classes and the actual data
  #and print metrics
  cm = confusion_matrix(testClassType, predictedClass)
  print(cm)
  tn, fp, fn, tp = cm.ravel()
  print("Accuracy", (tp+tn)/(tn+tp+fp+fn), "precision", tp/(tp+fp), "recall", tp/(tp+fn))
  
  print("Logistic Regression Classifier Result")
  #get the predicted classes from Scikit's logistic regression classifier
  predictedClass = logisticRegression(trainingFeatureData, testFeatureData, trainingClassType)
  #create the confusion matrix based on the predicted classes and the actual data
  #and print metrics
  cm = confusion_matrix(testClassType, predictedClass)
  print(cm)
  tn, fp, fn, tp = cm.ravel()
  print("Accuracy", (tp+tn)/(tn+tp+fp+fn), "precision", tp/(tp+fp), "recall", tp/(tp+fn))
  
if __name__ == '__main__':
  if len(sys.argv) > 1:
    main()
  else:
    print("Needs to provide spambase.csv as argument")