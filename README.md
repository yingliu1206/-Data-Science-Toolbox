# -Data-Science-Toolbox

**Background**

* Throughout this course you will be designing and implementing a Data Science Toolbox using Python. There will be 4 (or 5) major deliverables and required, supporting discussion posts. The first deliverable will be focused on designing the class hierarchy, building the basic coding infrastructure, and beginning the documentation process.

**Overview of Deliverables**
  1. Design
  2. Implement DataSet Class and subclasses
  3. Implement ClassifierAlgorithm Class, simpleKNNClassifier, and Experiment Class
  4. Implement ROC and kdTreeKNNClassifier
  5. ARM (greedy tree search ARM) OR hmmClassifier (dynamic programming) and final package

**Software Design Requirements Overview**

* The toolbox will be implemented using OOP practices and will take advantage of inheritance and polymorphism. Specifically, the toolbox will consist of 3 main classes some of which have subclasses and member methods as noted below. You will also submit a demo script for each submission that tests the capabilities of your newly created toolbox.
         
1. Class Hierarchy 
  a. DataSet
    i. TimeSeriesDataSet 
    ii. TextDataSet
    iii. QuantDataSet
    iv. QualDataSet 
    
  b. ClassifierAlgorithm
    i. simplekNNClassifier
    ii. kdTreeKNNClassifier
    iii. hmmClassifier 
    iv. graphkNNClassifier 
    
  c. Experiment
  
2. Member Methods for each Super and Sub Class (subclasses will have more specified members as well to be added later). Each subclass will inherit superclass constructor. All other member methods will be overridden unless design deviation is well-justified.
a. DataSet
  i. __init__(self, filename)
  ii. __readFromCSV(self, filename)
  iii. __load(self, filename)
  iv. clean(self)
  v. explore(self)
  
b. ClassifierAlgorithm
  i. __init__(self) 
  ii. train(self)
  iii. test(self) 

c. Experiment
  i. runCrossVal(self, k)
  ii. score(self)
  iii. __confusionMatrix(self)
 
