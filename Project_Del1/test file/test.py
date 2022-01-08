#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 22:26:00 2020

"""

import sys
sys.path.append(".")
from project_del1 import (DataSet, QuantDataSet, QualDataSet, TextDataSet, TimeSeriesDataSet)




def DataSetTests():
    print("DataSet Instantiation invokes both the __load() and the __readFromCSV() methods....")
    data = DataSet("survey.csv")
    print("==============================================================")
    print("Check DataSet member attributes...")
    print("DataSet.df:", data.df)

    print("==============================================================")
    print("Check DataSet member methods...\n")
    print("Instantiating the DataSet class again both the load() and the readFromCSV() methods run.")

    print("Now call DataSet.clean()...")
    data.clean(data.df)
    print("===========================================================")
    print("Now call DataSet.explore()...")
    data.explore()
    print("===========================================================")
    print("Now call DataSet.head()...")
    data.head(5)
    print("\n\n")

def QuantDataSetTests():
    quantdata = QuantDataSet("quant.csv")
    print("Check inheritence ...")
    quantdata.head(5)
    print("===========================================================")
    print("Check member methods...\n") 
    print("QuantDataSet.summaryStats():")      
    quantdata.summaryStats()
    print("QuantDataSet.correlation():")      
    quantdata.correlation()
    print("QuantDataSet.analysis():")   
    quantdata.analysis()
    print("===========================================================")
    print("Check that clean and explore methods have been overriden...\n")       
    print("QuantDataSet.clean():")
    quantdata.clean(quantdata.df)
    print("QuantDataSet.explore():")
    quantdata.explore()
    print("\n\n")
    
def QualDataSetTests():
    qualdata = QualDataSet("qual.csv")
    print("Check inheritence ...")
    qualdata.head(5)
    print("===========================================================")
    print("Check QualDataSet member attributes...")
    qualdata.getDim()
    qualdata.getType(1)
    qualdata.convertType(0, str)
    qualdata.valueCount(1)
    qualdata.replace(0,1)
    qualdata.sort(1)
    qualdata.oneHotEncoding(3)
    print("===========================================================")
    print("Check that clean and explore methods have been overriden...\n")
    print("QuanlDataSet.clean():")
    qualdata.clean()
    print("QuanlDataSet.explore():")
    qualdata.explore()
    print("\n\n")
    
def TextDataSetTests():
    textdata = TextDataSet("text.csv")
    print("Check inheritence ...")
    textdata.head(5)
    print("===========================================================")
    print("Check TextDataSet member attributes...")
    textdata.wordCount(1)
    textdata.stopwordRemove(0, 'English')
    textdata.tokenizer()
    textdata.stemming()
    textdata.topWords(5)
    textdata.Tfidf()
    print("===========================================================")
    print("Check that clean and explore methods have been overriden...\n")
    print("TextDataSet.clean():")
    textdata.clean()
    print("TextDataSet.explore():")
    textdata.explore()
    print("\n\n")
    
def TimeSeriesDataSetTests():
    TSdata = TimeSeriesDataSet("ts.scv","2020-09-20")
    print("Check inheritence ...")
    TSdata.head(5)
    print("===========================================================")
    print("Check TimeSeriesDataSet member attributes...")
    TSdata.dateWeek(TSdata.df)
    TSdata.findSeasonality()
    TSdata.frequency(TSdata.df)
    TSdata.statsCharateristics(TSdata.df)

    print("===========================================================")
    print("Check that clean and explore methods have been overriden...\n")
    print("TimeSeriesDataSet.clean():")
    TSdata.clean(TSdata.df)
    print("TimeSeriesDataSet.explore():")
    TSdata.explore()
    print("\n\n")

#=====================================================================
# Testing Classifier Class 
# (Not meant to be called, but will show instantiation, attributes,
# and member methods)
#=====================================================================
from project_del1 import (ClassifierAlgorithm, kdTreeKNNClassifier, SimpleKNNClassifier)
                                        
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
# Testing Classifier Class 
# (Not meant to be called, but will show instantiation, attributes,
# and member methods)
#=====================================================================
from project_del1 import Experiment

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

