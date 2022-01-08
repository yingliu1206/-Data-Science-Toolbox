#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append(".")
from project_del3 import (DataSet, QuantDataSet, QualDataSet, TextDataSet, TimeSeriesDataSet)



def DataSetTests():
    print("DataSet Instantiation invokes both the __load() and the __readFromCSV() methods....")
    data = DataSet("yelp.csv")
    print("==============================================================")
    print("Check DataSet member attributes...")
    print("DataSet.df:", data.df)
    print("==============================================================")
    print("Check DataSet member methods...\n")
    print("Instantiating the DataSet class again both the load() and the readFromCSV() methods run.")
    print("Now call DataSet.clean()...")
    data.clean()
    print("===========================================================")
    print("Now call DataSet.explore()...")
    data.explore()
    print("===========================================================")
    print("Now call DataSet.head()...")
    data.head(5)
    print("\n\n")

# DataSetTests()

    
def TextDataSetTests():
    textdata = TextDataSet("yelp.csv")
    print("Check inheritence ...")
    textdata.head(5)
    
    print("===========================================================")
    print("Check that clean and explore methods have been overriden...\n")
    print("TextDataSet.clean():")
    textdata.clean()
    print("TextDataSet.explore():")
    textdata.explore()    
    
    print("===========================================================")
    print("Check TextDataSet member attributes...")
    textdata.wordCount(0)
    textdata.stemming()
    textdata.topWords(5)
    textdata.Tfidf()
    textdata.vectorization()
    print("\n\n")

#TextDataSetTests()

def QuantDataSetTests():
    quantdata = QuantDataSet("Sales_Transactions_Dataset_Weekly.csv")
    print("Check inheritence ...")
    quantdata.head(5)
    print("===========================================================")
    print("Check member methods...\n") 
    print("QuantDataSet.correlation():")      
    quantdata.correlation()
    print("QuantDataSet.analysis():")   
    quantdata.analysis(quantdata.df)
    print("===========================================================")
    print("Check that clean and explore methods have been overriden...\n")       
    print("QuantDataSet.clean():")
    quantdata.clean(quantdata.df)
    print("QuantDataSet.explore():")
    quantdata.explore()
    print("\n\n")

# QuantDataSetTests()
    
def QualDataSetTests():
    qualdata = QualDataSet("loan_final.csv")
    print("Check inheritence ...")
    qualdata.head(5)
    print("===========================================================")
    print("Check QualDataSet member attributes...")
    qualdata.getDim()
    qualdata.getType()
    qualdata.convertType(0, 'str')
    qualdata.valueCount(1)
    qualdata.oneHotEncoding(['home_ownership'])
    print("===========================================================")
    print("Check that clean and explore methods have been overriden...\n")
    print("QuanlDataSet.clean():")
    qualdata.clean()
    print("QuanlDataSet.explore():")
    qualdata.explore('loan_amount','grade')
    print("\n\n")
    
# QualDataSetTests()
    
def TimeSeriesDataSetTests():
    TSdata = TimeSeriesDataSet("occupancy.csv")
    print("Check inheritence ...")
    TSdata.head(5)
    print("===========================================================")
    print("Check TimeSeriesDataSet member attributes...")
    TSdata.ParseDate()
    TSdata.findSeasonality('Humidity')
    TSdata.statsCharateristics()

    print("===========================================================")
    print("Check that clean and explore methods have been overriden...\n")
    print("TimeSeriesDataSet.clean():")
    TSdata.clean()
    print("TimeSeriesDataSet.explore():")
    TSdata.explore('Occupancy','Temperature')
    print("\n\n")

# TimeSeriesDataSetTests()

#=====================================================================
# Testing Classifier Class 
# (Not meant to be called, but will show instantiation, attributes,
# and member methods)
#=====================================================================
from project_del3 import (ClassifierAlgorithm, kdTreeKNNClassifier, SimpleKNNClassifier)
                                        
def ClassifierAlgorithmTests(X,y):
    print("ClassifierAlgorithm Instantiation....")
    classifier = ClassifierAlgorithm(X,y)
    print("==============================================================")
    print("Check ClassifierAlgorithm member attributes...")
    print("ClassifierAlgorithm.labels:", classifier.y)
    print("ClassifierAlgorithm.predictor:")
    print(classifier.X)
    print("==============================================================")
    print("Check class member methods...\n")
    print("ClassifierAlgorithm.train(data):")
    print(classifier.train(X,y))
    print("ClassifierAlgorithm.test(data):")
    print(classifier.test(y))
    print("===========================================================\n\n")

#ClassifierAlgorithmTests()
    
def simpleKNNClassifierTests(X,y):
    print("simpleKNNClassifier Instantiation....")
    KNN = SimpleKNNClassifier(5, X, y)

    print("==============================================================")
    print("Check member attributes...")
    KNN.split()
    print("simpleKNNClassifier.predictor:", KNN.X)
    print("simpleKNNClassifier.labels:", KNN.y)
    print("simpleKNNClassifier.k:",KNN.k)
    print("==============================================================")
    print("Check class member methods...\n")
    print("get simpleKNNClassifierAlgorithm X_train y_train X_test y_test:")
    X_train = KNN.get_X_train().values
    y_train = KNN.get_y_train().quality.values
    X_test = KNN.get_X_test().values
    y_test = KNN.get_y_test().quality.values
    print("simpleKNNClassifier.train:")
    KNN.train(X_train, y_train)
    print("simpleKNNClassifier.test:")
    y_pred = KNN.test(X_test)
    print("===========================================================\n\n")
    print(y_pred)

def kdTreeKNNClassifierTests():
    print("kdTreeKNNClassifier Instantiation....")
    classifier = kdTreeKNNClassifier(dataset = "c", k=3)
    print("==============================================================")
    print("Check member attributes...")
    print("kdTreeKNNClassifier.labels:", classifier.labels)
    print("kdTreeKNNClassifier.predictor (default should be none):")
    print(classifier.predictor)
    print("==============================================================")
    print("Check class member methods...\n")
    x = "data"
    print("kdTreeKNNClassifier.train(data):")
    print(classifier.train(x))
    print("kdTreeKNNClassifier.test(data):")
    print(classifier.test(x))
    print("===========================================================\n\n")
    print(classifier.y_pred)
    

#=====================================================================
# Testing Experiment Class 
# (Not meant to be called, but will show instantiation, attributes,
# and member methods)
#=====================================================================
from project_del3 import Experiment

def ExperimentTests():
    print("Experiment class instantiation...")
    simpleKNN = SimpleKNNClassifier(5, X, y)
    E = Experiment(X, y, [simpleKNN])

    print("==============================================================")
    print("Check class member methods...\n")
    print("Experiment.runCrossVal(numFolds):")
    E.runCrossVal(10)
    print("Experiment.score(test_labels, predicted_labels):")
    E.score(y_test, y_pred)
    print("Experiment.get_confusionMatrix(test_labels, predicted_labels):")
    E.get_confusionMatrix(y_test, y_pred)


    
def main():
    data = QuantDataSet('winequality-red.csv')
    data.clean()    
    X = data.get(0,11)
    y = data.get(11,12)
    ClassifierAlgorithmTests(X,y)
    simpleKNNClassifierTests(X,y)
    ExperimentTests()
    
if __name__=="__main__":
    main()

