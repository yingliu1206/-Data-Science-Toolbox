#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import os
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from wordcloud import WordCloud
#from nltk.tokenize import WhitespaceTokenizer
from nltk.stem.snowball import SnowballStemmer
from random import choices
import math
import operator

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
        super().__init__(filename)

    def clean(self):
        """
        Clean the quantitative data
        """
        super().clean()
        self.df = self.df.dropna()
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
    
    def get(self,slice1,slice2):
        return self.df.iloc[:,slice1:slice2]
    
    def get_values(self):
        return self.df.values 
    
    def get_labels(self):
        return self.df.columns.tolist()


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


class TransactionDataSet(DataSet):
    """ A dataset subclass for transaction data """


    def __init__(self, filename):
        """
        Create a new transaction dataset instance

        filename          the name of the dataset
        """
        # inherit from class Dataset
        super().__init__(filename) # read and load the csv data   


    def load_data_set(self):
        """
        Transfer the data frame to transaction data
        """
        transactions = []
        for i in range(0, 19):
            transactions.append([str(self.df.values[i,j]) for j in range(0, 4)])
        
        return transactions
                    
    def clean(self):
        """
        Clean the transaction data
        """
        transactions = self.load_data_set()
        cleanTransactions = [[i for i in nested if i != 'nan'] for nested in transactions]        
        
        return cleanTransactions

    def explore(self, supportThreshold = .25):
        """
        supportThreshold: the minimum support
        -----------
        Compute the Support, Confidence, and Lift for all Rules above supportThreshold by calling the private method __ARM__()
        """

        return self.__ARM__(supportThreshold = .25) 
    
    def __ARM__(self, supportThreshold = .25):
        """
        supportThreshold: the minimum support
        -----------
        Perform association rule mining.
        This is a priavte method and will be called by explore().
        """
        dataset = self.clean()
        
        # create the 1-item candidate by measuring the support_count of each item in the dataset
        C1 = list()
        for i in dataset:
            for j in i:
                C1.append(j)
        C1 = set(C1)
        
        # create item_set L1 by comparing C1 support_count with supportThreshold
        L1 = list()
        L1_tem = list()
        for i in C1:
            num = 0
            for j in dataset:      
                if(i in j):
                    num += 1
            #print("L1",i,'supportThreshold = ',num)
            if(num>=supportThreshold):
                L1.append(i)
            else:
                L1_tem.append([i])
                
        # create the 2-item candidate by using Item_set L1 from the previous step
        C2 = list()
        for i in L1:
            for j in L1:
                if(i<j):
                    C2.append(sorted([i,j]))
    
        # check all subsets in itemset are frequent if not, remove respective itemset from the list       
        C2_tem = C2.copy()
        for i in C2_tem:
            for n in L1_tem:
                if(set(n).issubset(set(i))):
                    print(i)
                    C2.remove(i)
    
        # create item_set L2 by comparing candidate_set C2 with supportThreshold
        L2 = list()
        L2_tem = list()
        for i in C2:
            num1 = 0
            for j in dataset:
                if(set(i).issubset(set(j))):
                    num1 +=1
            if(num1>=supportThreshold):
                #print('L2:',i,'supportThreshold =',num1)
                L2.append(i)
            else:
                L2_tem.append(i)
    
        # create the 3-item candidate by using Item_set L2 from the previous step
        C3 = list()
        for i in L2:
            for j in L2:
                length = len(i)
                set1 = set(i[0:length-1])
                set2 = set(j[0:length-1])
                result = list(set1.difference(set2))
                if(result==[] and list(set(i).difference(set(j)))!=[]):
                    C3_temp = set(i).union(set(j))
                    C3.append(sorted(list(C3_temp)))
        C3_df = pd.DataFrame(C3)
        C3 = C3_df.drop_duplicates().values.tolist()
        
        # check all subsets in itemset are frequent if not, remove respective itemset from the list       
        C3_tem = C3.copy()
        for i in C3_tem:
            for j in L2_tem:
                if(set(j).issubset(set(i))):
                    C3.remove(i)
        
        # create item_set L3 by comparing candidate_set C3 with supportThreshold
        L3 = list()
        for i in C3:
            num = 0
            for j in dataset:
                if set(i).issubset(j):
                    num += 1
            #print('L3',i,'supportThreshold = ',num)
            if num>=supportThreshold:
                L3.append(i)
                
        # create the association rules, the rules will be a list.
        # compute support and confidence
        result = list()
        for i in L2:
            for j in L3:
                if set(i).issubset(set(j)):
                    support = self.calsupport(j,i,dataset)
                    confidence = self.calconfidence(j,dataset)
                    sublist = list(set(j)-set(i))
                    result.append([i,sublist,support,confidence,support/confidence])
        
        return result

    def calsupport(self, itemset1,itemset2,dataset):
        """
        itemset1: the list for items whose support is larger than supportThreshold. 
                  Here we use item_set L3.
        itemset2: the list for items whose support is larger than supportThreshold. 
                  Here we use item_set L2.
        dataset: a list of Transactions, every transaction is also a list, which contains several items. 
        -----------
        Calculate the support
        """
        num1 = 0
        num2 = 0
        for i in dataset:
            if set(itemset1).issubset(i):
                num1 += 1
        for i in dataset:
            if set(itemset2).issubset(i):
                num2 += 1
        return num1/num2
    

    def calconfidence(self, itemset,dataset):
        """
        itemset: the list for items whose support is larger than supportThreshold. 
                 Here we use item_set L3.
        dataset: a list of Transactions, every transaction is also a list, which contains several items. 
        -----------
        Calculate the confidence
        """
        num1 = 0
        length = len(dataset)
        for i in dataset:
            if set(itemset).issubset(i):
                num1 += 1
        return num1/length

    def output(self):
        """
        Call the Rule class. 
        Display the top 10 rules along with these three measures to the console.
        """
        data = self.explore()
        r = Rule(data)
        
        return r.printRule()     
   
        
class Rule:
    """ A class to help with data organization"""    
    
    def __init__(self, data):
        """
        data: the association rules in TransactionDataSet
        -----------
        Run rule to help with data organization
        """
        self.data = data
    
    def printRule(self):
        aim = ['support', 'confidence', 'lift']
        for i in range(len(aim)):
            result = sorted(self.data, key = lambda x : x[i+2], reverse = True)[:10] 
            print("Top 10 rules for", aim[i], ":")
            for j in result:
                print(j[0],'==>',j[1],'Support ：',j[2],"Confidence : ",j[3],'Lift : ',j[4])
            print("=======================================================\n")
              
    
class ClassifierAlgorithm:

    '''
    Create Class for Classifier Algorithm
    '''

    def __init__(self, predictor, response):

        '''
        initialize the default attributes
        labels: integer       index of label column in the data
        predictor: integer    index of predictors columns in the data(optional)
        '''
        self.X = predictor
        self.y = response
        print('Init Successfully')
        

    def train(self,X,y):

        '''
        the default training method for classifier
        '''
        self.X = X
        self.y = y
        print('train successfully')

    def test(self,y):

        '''
        the default testing method for classifier
        '''
        y_pred = choices(y.iloc[:,0].unique(),k=len(y))        
        print('test successfully')
        return y_pred

    def __repr__(self):

        '''
        print readable result
        '''
        return '...'


class SimpleKNNClassifier(ClassifierAlgorithm):

    '''
    use the simple KNN classifier to analyze
    '''

    def __init__(self,k,predictor,response):

        '''
        initialize the default attributes
        labels: integer       index of label column in the data
        predictor: integer    index of predictors columns in the data(optional)
        '''
        super().__init__(predictor,response)
        self.eps = 1e-7 # T(n) = 1, S(n) = n(408)
        self.k = k      # T(n) = 1, S(n) = n(408)
        
        # T(n) = 1 +1 = 2
        # S(n) = 408 + 408 = 916
        # O(n) = 1
        
    def split(self):
        '''
        split the train and test dataset
        '''
        shuffle_X = self.X.sample(frac=1) # T(n) = 1, S(n) = n(408)
        shuffle_y = self.y.sample(frac=1) # T(n) = 1, S(n) = n(408)
        train_size = int(0.5*len(self.X)) # T(n) = 1, S(n) = n(286)
        self.X_train = shuffle_X[:train_size].reset_index(drop=True) # T(n) = 1, S(n) = n(286)
        self.y_train = shuffle_y[:train_size].reset_index(drop=True) # T(n) = 1, S(n) = n(286)
        self.X_test = shuffle_X[train_size:].reset_index(drop=True)  # T(n) = 1, S(n) = n(122)
        self.y_test = shuffle_y[train_size:].reset_index(drop=True)  # T(n) = 1, S(n) = n(122)
        print('Split dataset successfully')
        
        # T(n) = 7
        # S(n) = 1918
        # Big-O notation for T(n): O(1)
        # Big-O notation for S(n): O(1) 
    
    def get_X_train(self):
        return self.X_train # T(n) = 1, S(n) = n(286), O(n) = 1
    
    def get_X_test(self):
        return self.X_test  # T(n) = 1, S(n) = n(122), O(n) = 1
    
    def get_y_train(self):
        return self.y_train # T(n) = 1, S(n) = n(286), O(n) = 1
    
    def get_y_test(self):
        return self.y_test  # T(n) = 1, S(n) = n(122), O(n) = 1
    
    def train(self, X, y):
        self.X_train = X # T(n) = 1, S(n) = n(286), O(n) = 1
        self.y_train = y # T(n) = 1, S(n) = n(286), O(n) = 1
        print('train successfully')
        
    def test(self, X_test):
        distances = self.compute_distance(X_test) # T(n) = 1, S(n) = n(122)
        print('test function activates successfully')
        prediction, prob_list = self.predict_labels(distances)
        self.prob_list = prob_list
        self.prediction = prediction
        
        return prediction,prob_list
    
    def get_prob(self):
        '''
        get the probability of each labels
        '''
        ## O(n) = n^2
        ## S(n) = 3*408 + 408 + 2 = 1634
        
        matrix = self.prob_list
        prob_list = []
        for i in matrix:
            prob_dict = {}
            prob_dict.setdefault(5,0)
            prob_dict.setdefault(6,0)
            prob_dict.setdefault(7,0)
            values, counts = np.unique(i,return_counts=True)
            prob_dict.update(dict(zip(values,counts/5)))
            prob_list.append(max(prob_dict.values()))
        return prob_list
        
    

    def compute_distance(self, X_test):
        """
        To compute the euclidean distance for predictor and response
        """

        test_dim = X_test.shape[0]                         # T(n) =  1, S(n) = 1
        train_dim = self.X_train.shape[0]                  # T(n) = 1, S(n) = 1
        distances_matrix = np.zeros((test_dim, train_dim)) # T(n) = 1, S(n) = n^2

        for i in range(test_dim):                          # T(n) = n, S(n) = 408
            for j in range(train_dim):                     # T(n) = n, S(n) = 408
                distances_matrix[i, j] = np.sqrt(
                    self.eps + np.sum((X_test[i, :] - self.X_train[j, :]) ** 2))

        return distances_matrix
        
    # T(n) = n^2 + 3, S(n) = n^2 + 918
    # Big-O notation for T(n): O(n^2)
    # Big-O notation for S(n): O(n^2) 


    def predict_labels(self, distances):
        '''
        To find the probability for each label
        and select the label with highest probability
        '''
        test_dim = distances.shape[0]             # T(n) = 1, S(n) = 1
        y_pred = np.zeros(test_dim)               # T(n) = 1, S(n) = n^2
        K_neighbors = []
        for i in range(test_dim):                 # T(n) = n, S(n) = n
            index_y = np.argsort(distances[i, :]) # T(n),S(n) = n
            k_neighbors = self.y_train[index_y[: self.k]].astype(int) # T(n) = 1, S(n) = n
            y_pred[i] = np.argmax(np.bincount(k_neighbors))           # T(n) = n, S(n) = n
            K_neighbors.append(k_neighbors)
        return y_pred, K_neighbors
    
    # T(n) = 3 + 3n
    # S(n) = n^2 + 3n + 1
    # Big-O notation for T(n): O(n)
    # Big-O notation for S(n): O(n^2) 
    

def calcShannonEnt(dataset,dimen = -1): 
    '''
    to calculate the shannon entropy
    
    dataset: the input dataset
    
    dimen: rule to split dataset
    
    
    return value: shannon entropy
    '''

############### O(n) = n^2*log(n) #################
############### S(n) = 6(6*6 + 6*6*3*3) = 2160 ###### 

    numEntries = len(dataset) # get the dimension
    labelCnt = {}
    for currentLabel in dataset:
        if currentLabel[dimen] not in labelCnt.keys():
            labelCnt[currentLabel[dimen]] = 0
        labelCnt[currentLabel[dimen]] += 1
    shannonEnt = 0.0
    for key in labelCnt: # calculate the shannon entropy
        prob = float(labelCnt[key])/numEntries
        shannonEnt -= prob * math.log(prob,2)
    return shannonEnt

def splitDataSet(dataset,axis,value): 
    '''
    to split dataset by the rule
    
    dataset：matrix
    
    axis：value in the each row
    
    value：rule to split
    '''
    
    ############### O(n) = n^2 #################
    ############### S(n) = 36  #################
    
    retDataSet = []
    for featVec in dataset:
        if featVec[axis] == value: # filter the features
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataset):
    '''
    split by the least shannon entropy
    
    dataset：matrix
    '''
    
    ############### O(n) = n^2*log(n) + n^2*n*log(n)*n = n^3*log(n) ##########
    ############### S(n) = 4 + 6*408*4*2 = 19588                    ##########
    
    numFeatures = len(dataset[0]) - 1 
    baseEntropy = calcShannonEnt(dataset,dimen = -1)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [examples[i] for examples in dataset]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataset,i,value)
            prob = len(subDataSet)/float(len(dataset))
            newEntropy += prob*calcShannonEnt(subDataSet,dimen = -1) 
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature     

def majorityCnt(classList):
    '''
    Determination of the final label by majority vote
    '''
    
    ############### O(n) = n           #######
    ############### S(n) = 408*2 = 916 ####### 
    
    classCnt = {}
    for vote in classList:
        if vote not in classCnt.keys():
            classCnt[vote] = 0
        classCnt[vote] += 1
        
    # classCount.iteritems() decomposing the classCount dictionary into a tuple list.
    # operator.itemgetter(1) sorting the tuple in the order of the second element.
    
    sortedClassCnt = sorted(classCnt.items(), key = operator.itemgetter(1),reverse=True)
    return sortedClassCnt[0][0]

def createTree(dataset,labels):
    '''
    create the decision tree
    
    dataset：dataset in matrix format 
    
    labels：the whole labels except the target
    '''
    
    ############### O(n) = n^3*log(n) + n^2 + n^2 = n^3*log(n)    #################
    ############### S(n) = 408 + 408 + 408 + 19588 + 408 + 4 + 4988 = 21224 ####### 
    
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataset[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataset,bestFeat,value),subLabels)
    return myTree

def getNumLeafs(myTree):
    '''
    to get the number of leafs
    '''
    
    ############### O(n) = n    #################
    ############### S(n) = 408*4 = 1632 ######### 
    
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    '''
    get the dimension of the decision tree
    
    myTree：decision tree in dictionary format
    
    return value：the dimension
    '''
    
    ############### O(n) = n    #################
    ############### S(n) = 408*4 = 1632 ######### 
    
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def classify(inputTree,featLabels,testVec):
    '''
    decision classifier
    
    inputTree：Enter a decision tree classifier that has been trained with a structure of dictionary nested data.
    featlabels：A list of feature labels, the members of which are strings that represent each member of the next parameter in turn.
    testVector：A test vector without a classification label, but whose members represent the meaning of each feature that corresponds to the previous parameter.
    '''
    
    ############### O(n) = n^3          ###########
    ############### S(n) = 408*4 = 1632 ########### 
    
    valueOfFeat = inputTree
    while True:
        if isinstance(valueOfFeat, dict): 
            firstStr = list(valueOfFeat.keys())[0]
            secondDict = valueOfFeat[firstStr]
            featIndex = featLabels.index(firstStr)
            key = testVec[featIndex]
            try:
                valueOfFeat = secondDict[key]
            except KeyError:
                classLabel = 6
                break
        else: 
            classLabel = valueOfFeat
            break
    return classLabel

def storeTree(inputTree,filename):
    '''
    store the trained model
    '''
    
    ############### O(n) = n    #################
    ############### S(n) = 408  ################# 
    
    import pickle
    fw = open(filename, "wb")
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    '''
    grab the trained model
    '''
    
    ############### O(n) = n    #################
    ############### S(n) = 408  ################# 
    
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)

decisionNode = dict(boxstyle='sawtooth',fc = '0.8')
leafNode = dict(boxstyle='round4',fc='0.8')
arrow_args = dict(arrowstyle='<-')

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    '''
    plot one node
    '''
    createPlot.axl.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',
    xytext=centerPt,textcoords='axes fraction',
    va='center',ha='center',bbox=nodeType,arrowprops=arrow_args)

def plotMidText(ccntrPt,parentPt,txtStr):
    xMid = (parentPt[0]-ccntrPt[0])/2.0 + ccntrPt[0]
    yMid = (parentPt[1]-ccntrPt[1])/2.0 + ccntrPt[1]
    createPlot.axl.text(xMid,yMid,txtStr)

def plotTree(myTree,parentPt,nodeTxt):
    '''
    plot the tree
    '''
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xoff + (1.0 + float(numLeafs))/2.0/plotTree.totalW,plotTree.yoff)
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yoff = plotTree.yoff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xoff = plotTree.xoff + 1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xoff,plotTree.yoff),cntrPt,leafNode)
            plotMidText((plotTree.xoff,plotTree.yoff),cntrPt,str(key))
    plotTree.yoff = plotTree.yoff + 1.0/plotTree.totalD

def createPlot(inTree):
    '''
    to plot the tree in dictionary format
    '''
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    createPlot.axl = plt.subplot(111,frameon=False)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xoff = -0.5/plotTree.totalW
    plotTree.yoff = 1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()
    
class Tree(object):
    '''
    The class for TreeABC
    '''
    def print_DT(self,inTree):
        createPlot(inTree)
        
class DecisionTree(Tree, ClassifierAlgorithm):
    '''
    Inheritate from Tree and ClassifierAlgorithm
    '''
    def __init__(self,data,label):
        self.myData = data
        self.label = label
        # print(label)
        # print(self.label)
        self.predict_result = []
        self.train_data = data[:len(data)//2] # use half of dataset to train
        self.test_data = data[len(data)//2:] # use other half of dataset to check
        
    def get_X_train(self):
        return self.train_data
    
    def get_y_train(self):
        return self.label
    
    def get_X_test(self):
        return self.test_data
    
    def train(self,X,y):
        '''
        to train the decision tree model
        '''
        ############### O(n) = 1    #################
        ############### S(n) = 408  #################
        self.dt = createTree(X,y)
        print(self.dt)
    def test(self,y):
        '''
        test on the X_test and then plot the tree
        '''
        ############### O(n) = n^3 * log(n)                              #################
        ############### S(n) = 1632 + 21224 + 21224 + 408 + 408 = 44896  #################
        self.true_result = []
        for spline_data in y:
            predict_res = classify(self.dt,['citric acid', 'residual sugar', 'density', 'pH', 'sulphates', 'alcohol'],spline_data[:-1]) # Prediction
            true_res = spline_data[-1]
            self.predict_result.append(predict_res)
            self.true_result.append(true_res)
        print("Test done!")
        print('Prediction: ')
        print(self.predict_result)
        print('True Value: ')
        print(self.true_result)
    
    

class Experiment:

    def __init__(self, X, y, clfs):
        """
        X: predictors
        y: labels
        clfs: a list of classifiers
        -----------
        Run Experiment/test from prediction of ClassifierAlgorithm
        """
        self.X = X
        self.y = y
        self.clfs = clfs
        
    def runCrossVal(self, k):
        """
        k: interger, number of fold for cross validation
        -----------
        Run cross validation with k-folds
        Return a df containing all predicted labels
        """
        #split data into k folds
        allX = self.X.values.tolist()
        ally = self.y.values.tolist()

        k_folds_X = []
        k_folds_y = []
        foldsize = int(len(self.X)/k)

        for _ in range(k):
            eachfoldX = []
            eachfoldy = []
            while len(eachfoldX) < foldsize:
                idx = np.random.randint(len(allX))
                eachfoldX.append(allX.pop(idx))
                eachfoldy.append(ally.pop(idx))
            k_folds_X.append(eachfoldX)
            k_folds_y.append(eachfoldy)
        
        #loop train test split for k folds
        for i in range(k):
            X_test = k_folds_X[i]
            y_test = k_folds_y[i]
            X_train = []
            y_train = []
            for j in range(k):
                if j != i:
                   X_train.extend(k_folds_X[j])
                   y_train.extend(k_folds_y[j])
        
                    
            #run each classifiers, get predictions, and store all pred labels for each clf in df
            pred_list = []
            pred_df = pd.DataFrame()

            for clf in range(len(self.clfs)):
                current = self.clfs[clf]
                current.train(np.asarray(X_train), np.ravel(np.asarray(y_train)))
                y_pred, p = current.test(np.asarray(X_test))
                y_pred_lst = y_pred.tolist()
                pred_list.append(y_pred_lst)
                  
            
        pred_df = pd.DataFrame(pred_list).transpose()                    
        print(pred_list)
        return pred_list

        print( "Experiment runCrossVal method executed successfully")



    def score(self, y_test, y_pred):
        """
        y_test: This will be a list that contains test labels of each classifier. 
                (It only includes test labels of SimpleKNNClassifier this time.)
        y_pred: This will be a list that contains predicted labels of each classifier. 
                (It only includes predicted labels of SimpleKNNClassifier this time.)
        -----------
        Compute the accuracy of each classifier and present the result as a table.
        """
        score_list = []
        
        for i,j in zip(range(len(y_test)), range(len(y_pred))):
                self.y_test = y_test[i] # T(n) = 1, S(n) = n(408)
                self.y_pred = y_pred[j] # T(n) = 1, S(n) = n(408)
                
                # compute the accuracy score of SimpleKNNClassifier
                correct = 0                              # T(n) = 1
                for x in range(len(self.y_test)):        # T(n) = 4n_1(n=408 for this case), S(n) = n(408)
                    if self.y_test[x] == self.y_pred[x]: # T(n) = 3n_1
                        correct += 1                     # T(n) = 2*(0 to n_1), S(n) = 0 to 408(worst case)
                score = (correct/float(len(y_test)))
                score_list.append(score) # T(n) = 3, S(n) = 1 
        print("Accuracy scores of each classifier:", score_list)
                
                
        # present the result as a table format
        KNN_score = score_list[0]
        simpleTree_score = score_list[1] # T(n) = 1, S(n) = 1
        svm_score ="None"        # T(n) = 1, S(n) = 1
        hmm_score ="None"        # T(n) = 1, S(n) = 1
        d = {"simplekNN": KNN_score,
             "DecisionTree": simpleTree_score,
             "svm": svm_score,
             "hmm":hmm_score}                                        # T(n) = 4, S(n) = 1
        print ("{:<16} {:<16}".format('Classifier','accuracyScore')) # T(n) = 2
        for k, v in d.items():                                       # T(n) = 3n_2(n=4 for this case), S(n) = 2 
            accuracyScore = v                                        # T(n) = n_2
            print ("{:<16} {:<16}".format(k, accuracyScore))         # T(n) = 2n_2
        
        print( "Experiment score method executed successfully")      # T(n) = 1

# T(n) = 3 + n_1(4+3+2) + 3 + 9 + n_2(3+1+2) + 1
#      = 9n_1 + 6n_2 + 16, as we always have four classifiers, n_2 = 4
# T(n) = 9n + 40
# S(n) = 4n + 7

# The worst case is when predicted labels completely equal to the test labels, we have to do addition 408 times and save 408 space for the varable: correct.
# T(n) = 9 * 408 + 40 = 3712, when n is the number of test labels and equals to 408.
# S(n) = 4 * 408 + 7 = 1639        

# Big-O notation for T(n): O(n)
# Big-O notation for S(n): O(n)


    def __confusionMatrix(self, y_test, y_pred):
        """
        y_test: This will be a list that contains test labels of each classifier. 
                (It only includes test labels of SimpleKNNClassifier this time.)
        y_pred: This will be a list that contains predicted labels of each classifier. 
                (It only includes predicted labels of SimpleKNNClassifier this time.)
        -----------
        This is a priavte method and will be called by get_confusionMatrix().
        """
        confusionMatrix_list = []
        
        for k,l in zip(range(len(y_test)), range(len(y_pred))):
                self.y_test = y_test[k] # T(n) = 1, S(n) = n(408)
                self.y_pred = y_pred[l] # T(n) = 1, S(n) = n(408)
        
                labels = np.unique(self.y_test)                        # T(n) = 2, S(n) = n_1
                confusionMatrix = np.zeros((len(labels), len(labels))) # T(n) = 3, S(n) = n_1^2
                
                for i in range(len(labels)):                           # T(n) = n_1^2(n=6 for this case), S(n) = n_1
                    for j in range(len(labels)):                       # S(n) = n_1
                        count = 0                                      # T(n) = 1
                        for g,p in zip (self.y_test, self.y_pred):     # T(n) = n_2(n=408 for this case), S(n) = n_2
                            if (g==labels[i]) & (p==labels[j]): 
                                count += 1                             # T(n) = 1 to n_2/2, S(n) = 1 to n_2/2
                        confusionMatrix[i,j] = count                   # T(n) = 1               
                
                confusionMatrix_list.append(confusionMatrix)
        
        classifier_list = ["simplekNN", "DecisionTree"]
        for m,n in zip(classifier_list, range(len(confusionMatrix_list))):
            print("The confusion matrix of", m, "is")
            print(confusionMatrix_list[n])
            print("==============================================\n")


# T(n) = 7 + n_1^2(1 + n_2 + n_2/2 + 1)
#      = 7 + 2*n_1^2 + 1.5*n_2*n_1^2, n_1 is the the number of assigned labels and n_2 is the number of test or predicted labels.
# S(n) = n_1^2 + 3*n_1 + 1.5*n_2

# The worst case is when predicted labels and test labels completely equal to assigned labels, we have to do addition 204 times and save 204 space for the varable: correct.
# T(n) = 7 + 2*36 + 1.5*408*36 = 22111, when n_1 = 6, n_2 = 408
# S(n) = 36 + 18 + 1.5*408 = 666      

# Big-O notation for T(n): O(n^3)
# Big-O notation for S(n): O(n)       
   
    def get_confusionMatrix(self, y_test, y_pred):
        """
        y_test: This will be a list that contains test labels of each classifier. 
                (It only includes test labels of SimpleKNNClassifier this time.)
        y_pred: This will be a list that contains predicted labels of each classifier. 
                (It only includes predicted labels of SimpleKNNClassifier this time.)
        -----------
        Compute and display a confusion matrix by calling the private method __confusionMatrix()
        """
        return self.__confusionMatrix(y_test, y_pred) 
         
        print("Experiment __confusionMatrix method executed successfully") 
        


    def ROC_curve(self, clfs):
        '''
        Draw ROC curve for classifer
        compute TPR and FPR
        '''
        for c in clfs:
            #if c == simpleKNN:
        ############## S(n) = 361 ############## 
        ############## T(n) = 100*3*n = O(n)############## 

            c.split()
            X_train = c.get_X_train().values
            y_train = c.get_y_train().quality.values
            X_test = c.get_X_test().values
            y_test = c.get_y_test().quality.values
            c.train(X_train, y_train)
            y_pred, p = c.test(X_test)
            y_pred = y_pred.tolist()
            prob = c.get_prob()
            # Iterate thresholds from 0.0, 0.01, ... 1.0
            thresholds = np.arange(0.0, 1.01, .01)              #####space:101
            # iterate through all thresholds and determine fraction of true positives
            # and false positives found at this threshold
            labelclass = [5,6,7]                                #####space:3
            fpr_lst =[]             
            tpr_lst = []
                
            for j in range(len(labelclass)):
            
                fpr = []
                tpr = []
                P = countX(y_pred, labelclass[j])
                N = len(y_pred) - P
                
                for thresh in thresholds:
                    FP=0
                    TP=0
                    # loop and calculate each label's TPR and FPR (one versus all)
                    for i in range(len(prob)):
                        if (prob[i] > thresh):
                            if y_pred[i] == labelclass[j] and y_test[i] == labelclass[j]:
                                TP = TP + 1
                            if y_pred[i] == labelclass[j] and y_test[i] != labelclass[j]:
                                FP = FP + 1
                    fpr.append(FP/float(N))
                    tpr.append(TP/float(P))
                # save to lst    
                fpr_lst.append(fpr)
                tpr_lst.append(tpr)
            
            # compute average of TPR and FPR for all labels
            fpr_lst_df = pd.DataFrame(fpr_lst) 
            #fpr_avg = list(fpr_lst_df.mean (axis=0))
            tpr_lst_df = pd.DataFrame(tpr_lst) 
            #tpr_avg = list(tpr_lst_df.mean (axis=0))
            plt.title('ROC Curve for KNN')
            plt.plot(list(fpr_lst_df.loc[0]), list(tpr_lst_df.loc[0]),linestyle='solid')
            plt.plot(list(fpr_lst_df.loc[1]), list(tpr_lst_df.loc[1]),linestyle='solid')
            plt.plot(list(fpr_lst_df.loc[2]), list(tpr_lst_df.loc[2]),linestyle='solid')
            plt.legend(['label 5 vs (6,7)', 'label 6 vs (5,7)', 'label 7 vs (5,6)'], loc='upper right')
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.show()
   

def countX(lst, x): 
    '''
    count how many of specific labels in a lst
    '''
    count = 0
    for ele in lst: 
        if (ele == x): 
            count = count + 1
    return count

