#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append(".")
from project_del2 import (DataSet, QuantDataSet, QualDataSet, TextDataSet, TimeSeriesDataSet)



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

# TextDataSetTests()

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
from project_del2 import (ClassifierAlgorithm, kdTreeKNNClassifier, SimpleKNNClassifier)
                                        
def ClassifierAlgorithmTests():
    print("ClassifierAlgorithm Instantiation....")
    classifier = ClassifierAlgorithm(dataset = " ")
    print("==============================================================")
    print("Check ClassifierAlgorithm member attributes...")
    print("ClassifierAlgorithm.labels:", classifier.labels)
    print("ClassifierAlgorithm.predictor (default should be none):")
    print(classifier.predictor)
    print("==============================================================")
    print("Check class member methods...\n")
    x = "data"
    print("ClassifierAlgorithm.train(data):")
    print(classifier.train(x))
    print("ClassifierAlgorithm.test(data):")
    print(classifier.test(x))
    print("===========================================================\n\n")

def simpleKNNClassifierTests():
    print("simpleKNNClassifier Instantiation....")
    classifier = SimpleKNNClassifier(dataset = "b", k=1)
    print("==============================================================")
    print("Check member attributes...")
    print("simpleKNNClassifierAlgorithm.labels:", classifier.labels)
    print("simpleKNNClassifier.predictor (default should be none):")
    print(classifier.predictor)
    print("simpleKNNClassifier.k:",classifier.k)
    print("==============================================================")
    print("Check class member methods...\n")
    x = "data"
    print("simpleKNNClassifier.train(data):")
    print(classifier.train(x))
    print("simpleKNNClassifier.test(data):")
    print(classifier.test(x))
    print("===========================================================\n\n")

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
    


#=====================================================================
# Testing Experiment Class 
# (Not meant to be called, but will show instantiation, attributes,
# and member methods)
#=====================================================================
from project_del2 import Experiment

def ExperimentTests():
    print("Experiment class instantiation (Experiment(classifier,data))...")
    experiment = Experiment("classifier","data")
    print("==============================================================")
    print("Check member attributes...")
    print("Experiment._classifier:",experiment._classifier)
    print("Experiment._data:",experiment._data)
    print("==============================================================")
    print("Check class member methods...\n")
    print("Experiment.score():")
    experiment.score()
    print("Experiment.runCrossVal(numFolds):")
    experiment.runCrossVal(5)
    print("==============================================================")
    print("Experiment.ROC_curve(): (This also calls the private method Experiment.__confusionMatrix())")
    experiment.ROC_curve()
    
    
def main():
    DataSetTests()
    QuantDataSetTests()
    QualDataSetTests()
    TextDataSetTests()
    TimeSeriesDataSetTests()
    ClassifierAlgorithmTests()
    simpleKNNClassifierTests()
    kdTreeKNNClassifierTests()
    ExperimentTests()
    
if __name__=="__main__":
    main()

