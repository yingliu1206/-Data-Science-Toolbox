#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
sys.path.append(".")
from project_del4 import (DataSet, QuantDataSet, QualDataSet, TextDataSet, TimeSeriesDataSet, TransactionDataSet)



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
    
def TransactionDataSetTests():
    TransactionData = TransactionDataSet('HealthyBasketData2.csv') 
    print("Check inheritence ...")
    TransactionData.head(5)
    print("===========================================================")
    print("Check TransactionDataSet member attributes...")
    print("TransactionDataSet.load_data_set():")
    TransactionData.load_data_set()
    print("TransactionDataSet.clean():")
    TransactionData.clean()
    print("TransactionDataSet.explore():")
    TransactionData.explore()
    print("TransactionDataSet.output():")
    TransactionData.output()

# TransactionDataSetTests()

#=====================================================================
# Testing Classifier Class 
# (Not meant to be called, but will show instantiation, attributes,
# and member methods)
#=====================================================================
from project_del4 import (ClassifierAlgorithm, SimpleKNNClassifier, DecisionTree, storeTree)
                                        
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
    y_pred, p = KNN.test(X_test)
    print("===========================================================\n\n")
    print(y_pred)
    return y_pred,y_test

    
def DecisionTreeTests(data):
    print("DecisionTreeClassifierTests Instantiation....")
    myData,labels = data.get_values().tolist(),data.get_labels()[:-1]
    storeTree(myData,'myData')
    storeTree(labels,'labels')
    dtr = DecisionTree(myData,labels)
    print("==============================================================")
    print("Check class member methods...\n")
    X_train = dtr.get_X_train()
    y_train = dtr.get_y_train()
    X_test = dtr.get_X_test()
    print("DecisionTreeClassifier train and test:")
    #print(classifier.predictor)
    dtr.train(X_train,y_train) 
    dtr.test(X_test)
    print("DecisionTreeClassifier prediction result:")
    # prediction
    dtr_pred = dtr.predict_result
    print(dtr_pred)
    print("DecisionTreeClassifier print out tree:")
    dtr.print_DT(dtr.dt)
    print("===========================================================\n\n")
    return dtr_pred,dtr.true_result
# DecisionTreeTests()


#=====================================================================
# Testing Experiment Class 
# (Not meant to be called, but will show instantiation, attributes,
# and member methods)
#=====================================================================
from project_del4 import Experiment

def ExperimentTests(X, y, clfs):
    print("Experiment class instantiation...")
    E = Experiment(X, y, clfs)
    print("==============================================================")
    print("Check class member methods...\n")
    print("Experiment.runCrossVal(numFolds):")
    E.runCrossVal(10)
    print("==============================================================")
    print("Experiment.score(test_labels, predicted_labels):")
    E.score(y_list_test, y_list_pred)
    print("==============================================================")
    print("Experiment.get_confusionMatrix(test_labels, predicted_labels):")
    E.get_confusionMatrix(y_list_test, y_list_pred)
    print("Experiment.ROC_curve([classifier]):")
    E.ROC_curve(clfs)
    

        
if __name__=="__main__":
    TransactionDataSetTests()
    data = QuantDataSet('winequality.csv')
    data.clean() 
    X = data.get(0,6)
    y = data.get(6,7)
    ClassifierAlgorithmTests(X,y)
    y_pred,y_test = simpleKNNClassifierTests(X,y)
    dtr_pred,dtr_test = DecisionTreeTests(data)
    y_pred2 = np.array(dtr_pred)
    y_test2 = np.array(dtr_test)
    y_list_test = [y_test, y_test2]
    y_list_pred = [y_pred, y_pred2]
    simpleKNN = SimpleKNNClassifier(5, X, y)    
    ExperimentTests(X, y, [simpleKNN])


