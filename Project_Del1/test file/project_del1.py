#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 23:24:32 2020

"""

class DataSet:
    """ A dataset class """
    
    def __init__(self, filename):
        self.df = filename
    
    def __readfromCSV(self): #can use A._DataSet__readfromCSV() to call method
        """
        read file from directory
        """
        print("__read method executed successfully")
    
    def read(self):
        self.__read()
        
    def __load(self):
        """
        load and display file
        """        
        print( "__load method executed successfully")
    
    
    def clean(self, df):
        """ 
        Clean the dataset and remove NA values  
        """
        
        print( "clean method executed successfully" )
        
        
    def explore(self):
        """
        Create two data visualization displays
        """
        
        print( "explore method executed successfully" )
    
    def head(self, n):
        """
        show/print the first n rows of the data
        """
        print( "head method executed successfully" )
    

    

class TextDataSet(DataSet):
    """ A dataset subclass for Text Data """
    
    def __init__(self, filename, delim = ",", newLine = "\n", headers = True, language = 'English'):
        
        """
        Create a new text dataset instance
        
        filename          the name of the dataset
        delim             string splited with ","
        newLine           new line used by "\n"       
        headers           header is set to TRUE if and only if the first row contains one fewer field than the number of columns
        language          string: The default language is English
        """
        
        super().__init__(filename)
        self.lang = language
        
        
    
    def clean(self, columns = "all"):
        """
        Clean all columns of the dataset and remove NA values  
        """
        
        print( "TextDateSet clean method executed successfully")
    
    def explore(self, stopwords=[]):
        
        """
        Create two data visualization displays: word cloud, wordcount frequency histogram
        """
        
        print( "TextDateSet explore method executed successfully")
    
       
    
    def wordCount(self, index):
        """
        Count how many words in certain row/index
        index        words that are counted
        """        
        self.index = index
        
        print( "TextDateSet wordCount method executed successfully")
    
    def stopwordRemove(self, line, lang):
        """
        Removes any stopwords like “and”, “but”, etc. in certain sentence/line/row/index

        lang        language used
        """
        self.line = line
        self.lang = lang
        
        print( "TextDateSet stopwordRemove method executed successfully")
    
    def tokenizer(self):
        """
        Converts input text to streams of tokens, 
        where each token is a separate word, punctuation sign, 
        number/amount, date, e-mail, URL/URI, etc
        """
        print( "TextDateSet Tokenizer method executed successfully")
    
    
    def stemming(self):
        """
        Extracts the base form of the words by removing affixes from them
        """
        print( "TextDateSet Stemming method executed successfully")
    

    
    def topWords(self, n):
        """
        Return the highest frequncy n words
        """
        print("TextDateSet topWords method executed successfully")
    
    def Tfidf(self):
        """
        Return the term frequency–inverse document frequency value
        
        """
        print( "TextDateSet tfidf method executed successfully")


class TimeSeriesDataSet(DataSet):
    """ A dataset subclass for time series data """    
    
    def __init__(self, filename, datetimeFormat):   
        """
        Create a new time series dataset instance

        filename          the name of the dataset
        datetimeFormat    the format to present the date
        """
        super().__init__(filename)
        #self.datetimeFormat = filename.datetime(year=2020, month=9, day=20) # assume the date is 09/20/2020

    def dateWeek(self, df):
        """
        Get the day of the week 
        """
        #self.dateWeek = df.strftime('%A')
        
        print( "dateWeek function executed successfully"  )    
    
    def findSeasonality(self):
        """
        Get the season of the date 
        """
        
        print( "FindSeasonality function executed successfully") 

    def frequency (self, df):
        """
        Apply a frequency to the data
        """
        #self.frequency = df.asfreq('D', method='pad')
        
        print( "Frequency function executed successfully" )  

    def statsCharateristics (self, df):
        """
        Get the statistical charateristics of the datsets
        """
        self.statsCharateristics=[]        
        #for i, dfr in enumerate(df):
            #self.statsCharateristics.append(df[i].describe())
        
        print( "StatsCharateristics function executed successfully"  )  
    
    def clean(self, df):
        """
        Clean time series dataset
        """
        #df.drop([], axis=1, inplace=True) # drop the unnecessary columns
        #df.dropna(inplace=True)  # drop NaN Values

        print( "clean function executed successfully"  )      
    
    def explore(self):       
        """
        Visualize time series date, ...
        """
        
        print( "explore function executed successfully")


class QuantDataSet(DataSet):
    """ A dataset subclass for quantitative data """    
    
    def __init__(self, filename):   
        """
        Create a new quantitative dataset instance
        
        filename          the name of the dataset
        """ 
        super().__init__(filename)  
        
    def clean(self, df):
        """
        Clean quantitative data
        """
        #super().clean(df)        
        #self.df.columns = self.df.columns.str.replace() # normalize column names
        #self.df = self.df.dropna(subset=[]) # use dropna to remove rows missing  
        
        print( "clean function executed successfully")
        
    def explore(self):
        
        """
        Create approriate graphs to show the data density, relationship...
        """
        
        print( "explore function executed successfully")
    
    def summaryStats(self):
        """
        Get the summary statistics, like max, min, mean, variance...
        """
        
        print( "SummaryStats function executed successfully")
    
    def correlation(self):        
        """
        Get the r value to find a correlation between two columns
        """
        
        print( "correlation function executed successfully")
    
    def analysis(self):        
        """
        Analyze the dataset, like by t-test
        """
        
        print( "analysis function executed successfully" )   

    
    
class QualDataSet(DataSet):
    
    """ A dataset subclass for  Qualitative Data """
    
    def __init__(self, filename):
        super().__init__(filename)
        
    def clean(self, columns = "all"):
        """
        clean the data, such as remove NA, convert to lowercase letters, split numericals and text
        """
        print( "QualDataset clean method executed successfully")
    
    def explore(self):
        
        """
        Create two data visualization displays: bar chart, histogram, heatmap, pie chart
        """
        
        print( "QualDataset explore method executed successfully")    
    

    def __getitem__(self, j):
        """Return jth row of data"""
        
        return self.df[j]

        
    def getDim(self):
        """
        Return the shape/dimension of the data: how many rows and columns
        
        """
        print( "QualDataset getDim method executed successfully")
    
    def getType(self, col):
        """
        Return the data type(int, str, float) of certain column of data
        
        """
        print( "QualDataset getType method executed successfully")
 
    def convertType(self, col, datatype):
        """
        Convert the data type(int, str, float) of certain column of data to other types
        
        """
        
        self.datatype = datatype        
        print( "QualDataset convertType method executed successfully")
    
    def valueCount(self, col):
        """
        Return the frequency distribution of categories within the column
        """
        
        print( "QualDataset valueCount method executed successfully")
    
    def replace(self, col, other):
        """
        Replace one column with another column/data
    
        """
        
        print( "QualDataset replace method executed successfully")
    
    
    
    def sort(self, col, aces = True):
        """
        Sort and return column(s) by its row (defualt: acesending order)
        """
        
        print( "QualDataset sort method executed successfully")
        
    def oneHotEncoding(self, col):
        """
        Convert each category value into a new column 
        and assign a 1 or 0 (True/False) value to the column
        (something like pd.get_dummies())
        """
        
        print( "QualDataset oneHotEncoding method executed successfully")
        
    



class ClassifierAlgorithm:

    '''
    Create Class for Classifier Algorithm
    '''

    def __init__(self,k = 1, dataset_type = None,dataset = None, labels = -1, predictor = None):

        '''
        initialize the default attributes
        labels: integer       index of label column in the data
        predictor: integer    index of predictors columns in the data(optional)
        '''

        self.k = k
        self.dataset_type = dataset_type
        self.df = dataset
        self.labels = labels
        self.predictor = predictor

    def train(self,df):

        '''
        the default training method for classifier
        '''

        print('train successfully')

    def test(self,df):

        '''
        the default testing method for classifier
        '''

        print('test successfully')

    def __repr__(self):

        '''
        print readable result
        '''

        return '...'

class kdTreeKNNClassifier(ClassifierAlgorithm):

    '''
    Inherite kdTreeKNNClassifier to the ClassifierAlgorithm class
    '''

    def __init__(self, k = 1, dataset_type = None, dataset = None, labels = -1, predictor = None):

        '''
        initialize the default attributes
        k: integer            number of nearesr neighbors to determine classification
        labels: integer       index of label column in the data
        predictor: integer    index of predictors columns in the data(optional)
        '''

        super().__init__()

    def train(self,df,leaf_size=0,metrics='euclidean',sort_results=True):

        '''
        Train the kdTreeKNNClassifier model
        '''

        self.leaf_size = leaf_size
        self.metrics = metrics
        self.sort_results = sort_results
        print('start kdTreeKNNClassifier training')

    def test(self,df,return_distance = False):

        '''
        Test the kdTreeKNNClassifier model
        '''
        self.return_distance = return_distance
        print('kdTreeKNNClassifier test result is ...')

class SimpleKNNClassifier(ClassifierAlgorithm):

    '''
    use the simple KNN classifier to analyze
    '''

    def __init__(self, k = 3, dataset_type = None, dataset = None, labels = -1, predictor = None):

        '''
        initialize the default attributes
        k: integer  number of nearesr neighbors to determine classification
        labels: integer       index of label column in the data
        predictor: integer    index of predictors columns in the data(optional)
        '''

        super().__init__()
        

    def train(self, df, k = 3, algorithm = 'auto', metric = 'minkowski'):

        '''
        train the simple KNN model
        '''

        self.algorithm = algorithm
        self.metric = metric
        print('Start SimpleKNN training...')

    def test(self,df):

        '''
        test the simple KNN model
        '''

        print('SimpleKNN Test Result is ...')






class Experiment:
    
    """ Run Experiment/test from prediction of ClassifierAlgorithm """
    def __init__(self, classifier, data):
        
        self._classifier = classifier
        self._data = data
    
    def runCrossVal(self, k):
        """
        k: interger, number of fold for cross validation
        -----------
        Run cross validation with k-folds
        """
        
        print( "Experiment runCrossVal method executed successfully")
    
    def score(self):
        """
        Return prediction scores such as recall, precision, F1 score
        """
        print( "Experiment score method executed successfully")
    
    def __confusionMatrix(self):
        """
        Return the confusion matrix
        """
        
        print("Experiment __confusionMatrix method executed successfully")
        
    def ROC_curve(self):
        """
        return the ROC curve and confusion matrix
        
        """
        
        self.__confusionMatrix()
        print("Experiment ROC_curve method executed successfully")
        
       
    
    

    
        
        