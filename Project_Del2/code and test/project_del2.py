#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import os
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
#from nltk.tokenize import WhitespaceTokenizer
from nltk.stem.snowball import SnowballStemmer
from datetime import datetime

path = os.getcwd()
os.chdir(path)
print('The work enviroment is: ',os.getcwd())

class DataSet:
    """ A dataset class """
    
    def __init__(self, filename):
        print('The file name is: ' + filename)
        self.category = input("What's the type of dataset: \n")
        self.df = self.__load(filename)

 
    
    def __readfromCSV(self,filename): 
        """
        read file from directory when file is csv format
        """
        self.df = pd.read_csv(filename)

        return self.df
        print("__readfromCSV method executed successfully")
    
    def __readfromJSON(self,filename):
        """
        read file from directory when file is json format
        """
        self.df = pd.read_json(filename)
        return self.df
        print("__readfromJSON method executed successfully")
    
    def __load(self,filename):
        """
        load csv or json file
        """ 
        if (filename.endswith('.csv')):
            self.df = self.__readfromCSV(filename)
            print('load csv file successfully')
            return self.df
        elif (filename.endswith('json')):
            self.df = self.__readfromJSON(filename)
            print( "load JSON file successfully")
            return self.df
        else:
            print('Error Message: please input dataset format in csv or json')
        
    def clean(self):
        """ 
        Clean the dataset and drop duplicates
        """
        self.df = self.df.drop_duplicates()
        print( "clean method executed successfully" )
        
        
    def explore(self):
        """
        Display number of rows and columns, column names, and descriptive information
        Create visualizations of data

        """
        nrow = self.df.shape[0]
        ncol = self.df.shape[1]
        colname = self.df.columns
        descriptive = self.df.describe()
        category = self.category
        
        print('The number of rows: ',nrow)
        print('The number of columns: ',ncol)
        print('The column name: ',colname)
        print('Descriptive Information: ',descriptive)
        
        if category == 'quantitative' or category == 'qualitative' or category == 'timeseries' :
            # Create a scatterplot to look at the relationship between column 1 and column 2
            plt.scatter(self.df[self.df.columns[0]], self.df[self.df.columns[1]], alpha=0.5) 
            plt.title('Scatterplot of first column and second column')
            plt.xlabel('first column')
            plt.ylabel('second column')
            plt.show()
    
            # Create a line graph to look at the relationship between column 3 and column 4     
            plt.plot(self.df[self.df.columns[2]], self.df[self.df.columns[3]])
            plt.title('Line graph of third column and fourth column')
            plt.xlabel('third column')
            plt.ylabel('fourth column')
            plt.show()

                                  
        if category == 'text':
            #bar chart of star ratings
            plt.figure(figsize=(8,4))
            self.df['stars'].value_counts().plot(kind='bar',alpha=0.5,color='green',label='ratings',title='Count of Star Ratings')
            plt.legend()
            plt.show()
            
        print( "explore method executed successfully" )
    
    def head(self, n):
        """
        show the first n rows of the data
        n     the input row index 
        """
        headrows = self.df.head(n)
        print(headrows)
        print( "head method executed successfully" )
    

   
class TextDataSet(DataSet):
    """ A dataset subclass for Text Data """
    
    def __init__(self, filename, language = 'english'):
        
        """
        Create a new text dataset instance
        
        filename          the name of the dataset
        language          string: The default language is English
        """
        
        super().__init__(filename)
        self.lang = language
        
    def clean(self):
        """
        Clean columns of the dataset
        remove useless columns
        add a new column - "text_length" denotes the number of words in each review        
        remove punctuation and stopwords
        tokenization
        """
        self.df = self.df[['stars','cool','useful','funny','text']]
        #add a column containing text length
        list_review=[]
        for i in self.df['text']:
            list_review.append(len(i))
        self.df['text_length']=list_review
        #remove punctuation and stopwords
        self.df['text'] = self.punctuationRemove(self.df['text'])
        self.df['text'] = self.stopwordRemove(self.df['text'])

       
        print( "TextDateSet clean method executed successfully")
   
    def wordCount(self, j):
        """
        Count how many words in certain row/index
        j        the index jth row
        """        
        self.j = j
        count = len(self.df['text'][j])
        print("Number of words in this row:",count)
        #print( "TextDateSet wordCount method executed successfully")
    
    def stopwordRemove(self, serie):
        
        """
        Removes any stopwords such as “and”, “but”, and tokenization
        serie    input list or column
        """
        
        tokenizer = nltk.WordPunctTokenizer()
        self.result_serie= []
        stop_words = set(stopwords.words(self.lang))
        for row in serie:
            aux = []            
            text_row = tokenizer.tokenize(row.lower())
            for word in text_row:
                if word not in stop_words: # stopwords
                    aux.append(word)
            self.result_serie.append(' '.join(aux))
        return self.result_serie        
        #print( "TextDateSet stopwordRemove method executed successfully")
    
    def punctuationRemove(self, serie):
        
        """
        Removes any punctuations such as !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
        serie    input list or column
        """
        
        self.temp = []
        punctuation = [word for word in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~']
        punctuation += ['...', '  ', '\n']
        for element in serie:
            for word in punctuation:
                element = element.replace(word,' ')
            self.temp.append(element)
        return self.temp
    
    def topWords(self, n):
        """
        Return the highest frequncy n words
        n     the number of top highest frequncy
        """
        self.topwords = self.df.text.str.split(expand=True).stack().value_counts()[0:n]
        print(self.topwords)
        
        #print("TextDateSet topWords method executed successfully")

    def explore(self):
        
        """
        Create three data visualization displays: 
        bar chart: count of star ratings, 
        word cloud, 
        highest 10 wordcount frequency histogram
        """
        #bar chart
        plt.figure(figsize=(8,4))
        self.df['stars'].value_counts().plot(kind='bar',alpha=0.5,color='green',label='ratings',title='Count of Star Ratings')
        plt.legend()
        plt.show()
        
        #word cloud
        review_cloud=WordCloud(width=600,height=400).generate(" ".join(self.df['text']))
        plt.figure(figsize=(10,8),facecolor='k')
        plt.imshow(review_cloud)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.show()
        
        #top 10 most frequent words
        self.topwords = self.df.text.str.split(expand=True).stack().value_counts()[0:10]
        plt.figure(figsize=(8,4))
        self.topwords.plot(kind='bar',alpha=0.5,color='orange',label='ratings',title='Frequncy of 10 Top Words')
        plt.legend()
        plt.show()
        
        print( "TextDateSet explore method executed successfully")
           
    def stemming(self):
        """
        Extracts the base form of the words by removing affixes from them
        """
        stemmer = SnowballStemmer(self.lang)
        self.df['text_sep'] = self.df['text'].str.split()
        self.df['stemmed'] = self.df['text_sep'].apply(lambda x: [stemmer.stem(y) for y in x])
        print('Show first 5 rows of stemmed results:\n', self.df['stemmed'][0:5])
        #print( "TextDateSet Stemming method executed successfully")
            
    def Tfidf(self):
        """
        Return the term frequency–inverse document frequency value
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        v = TfidfVectorizer()
        x = v.fit_transform(self.df['text'])
        self.m = x.toarray()
        print('Tfidf sparse matrix is:\n', self.m)
        return self.m
        print( "TextDateSet tfidf method executed successfully")

    def vectorization(self):
        """
        Vectorization of text, also return bag of words
        """
        from sklearn.feature_extraction.text import CountVectorizer
        vectorize = CountVectorizer()
        self.X = vectorize.fit_transform(self.df.text)
        bagofwords = len(vectorize.get_feature_names())
        print('How many features (bag of words): ', bagofwords)
        return self.X
        print( "TextDateSet vectorization method executed successfully")


class TimeSeriesDataSet(DataSet):
    """ A dataset subclass for time series data """    
    
    def __init__(self, filename):   
        """
        Create a new time series dataset instance

        filename          the name of the dataset
        datetimeFormat    the format to present the date
        """
        super().__init__(filename)

    def ParseDate(self):
        """
        Get the day of the week 
        """
        date_name = input("Please type your datetime's column name")
        self.df[date_name] = self.df[date_name].astype('datetime64[ns]')
        ans = input('what datetime type you want to parse?\n1.(year,month,day)\n2.(year,month,day,hour and second)')
        if ans == 1:
            self.df['year'] = self.df[date_name].dt.year
            self.df['month'] = self.df[date_name].dt.month
        if ans == 2:
            self.df['year'] = self.df[date_name].dt.year
            self.df['month'] = self.df[date_name].dt.month
            self.df['day'] = self.df[date_name].dt.day
            self.df['hour'] = self.df[date_name].dt.hour
            self.df['second'] = self.df[date_name].dt.second
        
        self.df.rename(columns={date_name:'date'})
        print( "ParseDate function executed successfully"  )    
    
    def findSeasonality(self,col):
        """
        Get the season of the date 
        """
        new_df = self.df[[col]]
        new_df.index = self.df['date']
        new_df.diff().plot(figsize=(20,10), linewidth=5, fontsize=20)
        plt.title('Plot first order difference of ' + col)
        plt.show()
        
        
        pd.plotting.autocorrelation_plot(new_df)
        plt.title('Plot of Autocorrelation')
        print( "FindSeasonality function executed successfully")  

    def statsCharateristics (self):
        """
        Get the statistical charateristics of the datsets
        """      
        print(self.df.describe())
        
        print( "StatsCharateristics function executed successfully"  )  
    
    def clean(self):
        """
        Clean time series dataset
        """
        
        self.df.dropna(inplace=True)  # drop NaN Values

        print( "clean function executed successfully"  )      
    
    def explore(self,target,col1):       
        """
        Visualize time series date, ...
        """
        value_counts = self.df[target].value_counts()
        
        labels = ['Occupied','Not Occupied'] 
  
        counts = [value_counts.iloc[0],value_counts.iloc[1]] 

        fig = plt.figure(figsize =(10, 7)) 
        
        plt.pie(counts, labels = labels,autopct='%1.1f%%') 
        plt.title('The distribution of target variable')
        
        plt.show()
        
        new_df = self.df[[col1]]
        
        new_df.index = self.df['date']
        
        fig,ax = plt.subplots()
        new_df.plot(ax=ax)
        
        ax.xaxis_date()

        #Make space for and rotate the x-axis tick labels
        fig.autofmt_xdate()
        
        
        print('the correlaiton between each variables is: ')
        print(self.df.corr())
        print( "explore function executed successfully")


class QuantDataSet(DataSet):
    """ A dataset subclass for quantitative data """    
    
    def __init__(self, filename):   
        """
        Create a new quantitative dataset instance
        
        filename          the name of the dataset
        """ 
        super().__init__('Sales_Transactions_Dataset_Weekly.csv')  
        
    def clean(self, df):
        """
        Clean the quantitative data
        """
        super().clean()        
        self.df = self.df.fillna(df.mean()) # fill in missing values with the mean
        print( "clean function executed successfully")
        
    def explore(self):
        
        """
        Explore the dataset by getting the number of rows and columns, column names, and descriptive information.
        Explore by visualizations.
        """        
        super().explore() 

        # Create a histogram of W1
        plt.hist(self.df.W1, bins = 10)
        plt.title('Histogram of W1')
        plt.xlabel('range')
        plt.ylabel('count')
        plt.show()
        
        # Create a boxplot for W1 to W4
        data = [self.df.W1, self.df.W2, self.df.W3, self.df.W4] 
        plt.boxplot(data)
        plt.title('Boxplot for W1 to W4')
        plt.xlabel('Weeks')
        plt.show()
            
    def correlation(self):        
        """
        Get the r value to find a correlation between two columns
        """
        correlation_matrix = np.corrcoef(self.df.W0, self.df.W1)
        correlation_FP = correlation_matrix[0,1]
        results = correlation_FP  
        print(results)

        print( "correlation function executed successfully")
    
    def analysis(self, df):        
        """
        Analyze the dataset by getting the sum of each row and each column.
        Sort the data frame by a specific column value.
        """
        Rowsums = df.sum(axis=1) # do a row sum
        Colsums = df.sum(axis=0) # do a column sum
        Sortvalue = df.sort_values(by='W0', ascending=False).head() # sort the value by a specific column
        
        print('The sum of rows: ',Rowsums)
        print('The sum of columns: ',Colsums)
        print('The data frame sorted by a specific column: ',Sortvalue)

        print( "analysis function executed successfully" )   

        
class QualDataSet(DataSet):
    
    """ A dataset subclass for  Qualitative Data """
    
    def __init__(self, filename):
        super().__init__(filename)
        
    def clean(self, columns = "all"):
        """
        clean the data, such as remove NA, convert to lowercase letters, split numericals and text
        """
        super().clean()
        print('start removing null values')
        
        ans = input('what do you want for dealing with null values: 1. mode 2. drop')
        if ans == '2':        
            self.df = self.df.dropna(axis=0).reset_index(drop=True)
        if ans == '1':
            fill_mode = lambda col: col.fillna(col.mode())
            self.df = self.df.apply(fill_mode, axis=0)
        # clean the outliers
        
        int_or_float_columns = self.df.loc[:, (self.df.dtypes == 'float') | (self.df.dtypes == 'int')].columns.tolist()
        print("before remove outliers, the dataset shape is: ", self.df.shape)
        for cols in int_or_float_columns:
            Q1 = self.df[cols].quantile(0.25)
            Q3 = self.df[cols].quantile(0.75)
            IQR = Q3 - Q1
            self.df[cols] = self.df[cols][~((self.df[cols] < (Q1 - 1.5 * IQR)) |(self.df[cols] > (Q3 + 1.5 * IQR)))]
            self.df.dropna(axis=0,inplace=True)
            self.df.reset_index(inplace=True,drop=True)
        
        print('after remove outliers, the dataset shape is: ',self.df.shape)
        
        
        print( "QualDataset clean method executed successfully")
    
    def convertType(self, col, datatype):
        """
        Convert the data type(int, str, float) of certain column of data to other types
        
        """
        chosen_col = input('Do you want to type index or column name?(1:Index, 2:Column Name)')
        if chosen_col == '1':
            self.df.iloc[:,col] = self.df.iloc[:,col].astype(datatype)
        else:
            self.df.loc[col] = self.df.loc[col].astype(datatype)
        print( "QualDataset convertType method executed successfully")
    
    def explore(self,num_target,cat_target):
        
        """
        Create two data visualization displays: bar chart, histogram, heatmap, pie chart
        """
        
        # plot for numerical variable
        Y = self.df[num_target]
        fig, axs = plt.subplots(1, 2, tight_layout=True)
        N, bins, patches = axs[0].hist(Y)
        
        # We'll color code by height, but you could use any scalar
        fracs = N / N.max()
        
        # we need to normalize the data to 0..1 for the full range of the colormap
        norm = colors.Normalize(fracs.min(), fracs.max())
        
        # Now, we'll loop through our objects and set the color of each accordingly
        for thisfrac, thispatch in zip(fracs, patches):
            color = plt.cm.viridis(norm(thisfrac))
            thispatch.set_facecolor(color)
        
        # We can also normalize our inputs by the total number of counts
        axs[1].hist(Y, density=True)
        
        # Now we format the y-axis to display percentage
        axs[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
        plt.show()
        
        self.df[cat_target].value_counts().plot(kind='bar')
        
        
        print( "QualDataset explore method executed successfully")    
    

    def __getitem__(self, j):
        """Return jth row of data"""
        return self.df.iloc[j]

        
    def getDim(self):
        """
        Return the shape/dimension of the data: how many rows and columns
        
        """
        print('The shape of cleaned dataset is: ',self.df.shape)
        print( "QualDataset getDim method executed successfully")
    
    def getType(self):
        """
        Return the data type(int, str, float) of certain column of data
        
        """
        print('The dtypes for selected column is: ',self.df.dtypes)
        print( "QualDataset getType method executed successfully")

    
    def valueCount(self, col):
        """
        Return the frequency distribution of categories within the column
        """
        chosen_col = input('Do you want to type index or column name?(1:Index, 2:Column Name)')
        if chosen_col == '1':
            print('The value count in ',col,' is: ',self.df.iloc[:,col].value_counts().sort_values(ascending=False))
        else:
            print('The value count in ',col,' is: ',self.df.loc[col].value_counts().sort_values(ascending=False))
        
        print( "QualDataset valueCount method executed successfully")
      
    def oneHotEncoding(self, col_list):
        """
        Convert each category value into a new column 
        and assign a 1 or 0 (True/False) value to the column
        (something like pd.get_dummies())
        """
        
        self.df = pd.get_dummies(self.df,columns=col_list)
        
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
        
       
    