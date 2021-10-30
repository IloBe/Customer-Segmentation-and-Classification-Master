
# Multi-Collinearity: https://stackoverflow.com/questions/25676145/capturing-high-multi-collinearity-in-statsmodels
# Imputation: https://www.theanalysisfactor.com/multiple-imputation-in-a-nutshell/
# Visualisation: catscatter for categoricals: https://towardsdatascience.com/visualize-categorical-relationships-with-catscatter-e60cdb164395

# Missing Values and Imputation: https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4
# Multi-Imputation with mice in Python: https://stackoverflow.com/questions/50670080/mice-implementation-in-python
#   df_train_numeric = df_train[['Age']].select_dtypes(include=[np.float]).as_matrix()
#   df_complete=MICE().complete(df_train_numeric)
#   with link: https://stackoverflow.com/questions/45239256/data-imputation-with-fancyimpute-and-pandas
# for pip: https://pypi.org/project/fancyimpute/
# https://github.com/Ouwen/scikit-mice  aufgrund des papers: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074241/  
#           enthält eine py Datei für den MiceImputer und einer öffentliche Methode transform() für LinearRegression model
# https://datascience.stackexchange.com/questions/19840/rs-mice-imputation-alternative-in-python
# Mice imputation with statsmodels library: https://www.statsmodels.org/dev/imputation.html
# statsmodels mice imputation: https://datascience.stackexchange.com/questions/52927/advice-on-imputing-temperature-data-with-statsmodels-mice?rq=1
# https://www.statsmodels.org/dev/generated/statsmodels.imputation.mice.MICEData.html#statsmodels.imputation.mice.MICEData
# https://pypi.org/project/fancyimpute/
# https://stackoverflow.com/questions/45321406/missing-value-imputation-in-python-using-knn

# imputation on missing values: https://towardsdatascience.com/missing-data-and-imputation-89e9889268c8

###
#During install of fancyimpute huge error message block appeared , ended with:
# ...
#C:\anaconda\anaconda3\envs\DS-arvato-project\lib\site-packages\numpy\distutils\system_info.py:1730: UserWarning:
#      Lapack (http://www.netlib.org/lapack/) sources not found.
#      Directories to search for the sources can be specified in the
#      numpy/distutils/site.cfg file (section [lapack_src]) or by setting
#      the LAPACK_SRC environment variable.
#    return getattr(self, '_calc_info_{}'.format(name))()
#  error: Microsoft Visual C++ 14.0 is required. Get it with "Build Tools for Visual Studio": https://visualstudio.microsoft.com/downloads/
#  {}
#  {}
#  ----------------------------------------
#  ERROR: Failed building wheel for scs
#ERROR: Could not build wheels for cvxpy which use PEP 517 and cannot be installed directly


###########################################
#
# import libraries
#
###########################################
import pandas as pd
import numpy as np
import scipy.stats as st
import collections
import datetime
import missingno as msno
from subprocess import call


# for ETL and ML model parts
from sklearn.experimental import enable_iterative_imputer, enable_hist_gradient_boosting
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, roc_curve


# for visualisation
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import display # Allows the use of display() for DataFrames
import seaborn as sns
sns.set('notebook')

# Suppress matplotlib user warnings
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")


###########################################
#
# EDA part
#
# functions list needed to modify datasets
###########################################

def map_to_nan(df, clm, vl):
    '''
    Returns a dataframe with column values mapped to NaN (empty values).
    
    Input:
        df - the dataframe to change
        clm - (string) the columnname the values are in that shall be mapped to NaN
        vl - the specific value to change (can be an int, float and a general string like 'XX')
    '''
    #print('clm: {}'.format(clm))
    #print('val: {}'.format(vl))   
    
    if clm in df.columns:
        df[clm].replace(vl, np.nan, inplace=True)
    
    return df


def is_nan(x):
    ''' 
    Returns boolean TRUE or FALSE, if the x param is an empty value working for
    Python's non-unique NaN and Numpy's unique NaN (singleton) resp. NaT for datetime objects;
    any other objects like e.g. string does not raise exceptions if encountered
    '''
    return (x is np.nan or x != x or (x == 'NaT') or ((type(x) is datetime.datetime) and (np.isnat(np.datetime64(x)))))  # and instead or; prove 'NaT' explicitly


def modify_CAMEO_DEU_2015(val):
    '''
    Returns the converted float value or the np.nan value for empty value based on the given param 'val'.
    
    Description:
    this function 'modify_CAMEO_DEU_2015()' shall only be used for associated dataframe column map() function.
    '''    
    if is_nan(val):
        return np.nan
    else: # remove with slicing
        return float(val[:1])


def modify_EINGEFUEGT(val):
    '''
    Returns the converted integer value or the np.nan value for empty value based on the given param 'val'.
    
    Description:
    this function 'modify_EINGEFUEGT()' shall only be used for associated dataframe column map() function.
    '''
    
    if is_nan(val):
        return np.nan
    else: # if not NaN or NaT
        return float(val)


# for OST_WEST_KZ feature: see:
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html?highlight=replace#pandas.DataFrame.replace
def replace_OST_WEST_KZ(df):
    '''
    Replaces the strings 'O' for East to 0, 'W' for West to 1 and -1 for unknown to NaN;
    numbers are added as float
    
    Input:
        df - dataframe including the feature 'OST_WEST_KZ' to handle the replacement
    Output:
        df - the modified dataframe
    '''
    
    dict_O_W_unknown = {'OST_WEST_KZ': {'O': 0, 'W': 1, '-1': np.nan}}
    df = df.replace(to_replace=dict_O_W_unknown)
    
    return df


def check_labels(list_obj=[], name='empty'):
    '''
    Checks if any string of the given list is empty or are blank strings.
    
    Input:
        list - list of strings
        name - name of the list, that shall be checked
    Output:
        result is printed
    '''
    # toDo: more general approach for blank string sequences
    res = ('' or ' ' or '  ' or '   ' or '    ' or '[]') in list_obj
    print("Is any string empty in {} label list? : {}".format(name, str(res)))


def get_empty_values_info(df=None, name='', component='column', thresh=25):
    '''
    Prints the number and percentage of emtpy values of each dataframe column and
    collects the columns having an amount by default of NaNs >25%.
    
    Input:
        df - dataframe to be checked for NaN or if column values of type datatime for NaT values
        name - name of the dataframe, that shall be investigated, default ''
        component - column or row the calculation shall happen on, default 'column'
        thresh - threshhold given in percentage, default 25%
    Output:
        dict - dictionary including all the column items with default >25% and their calculated NaN information.
    '''
    dict_over25perc = {}
    dict_less26perc = {}
    df_samples = df.shape[0]
    counter = 1
    
    try:
        if component == 'column':
            list_clms = set(df.columns.to_list())
            print("\033[1mAmount of {} columns included NaN/NaT values:\033[0m".format(name))
            for col in list_clms:
                sum_na_col = sum(df[col].isna())
                perc_na_col = sum_na_col*100/df_samples
                if perc_na_col > thresh:
                    dict_over25perc[col] = [sum_na_col, np.round(perc_na_col, decimals=2)]
                else:    # perc_na_col < 26
                    dict_less26perc[col] = [sum_na_col, np.round(perc_na_col, decimals=2)]
                    
                print("{}. '{}' includes: {}, means: \033[1m{:.4}%\033[0m".
                      format(counter, col, sum_na_col, perc_na_col))
                counter += 1
        else:  # component is 'row'
            # iteration over each dataframe row is too slow;
            
            # iterating over multiple columns via list comprehension and function f
            # https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions         
            # result = [f(row[0], ..., row[n]) for row in df[['col1', ...,'coln']].values]
            # or 
            # with given threshhold: thresh int, optional; require that many non-NA values.
            #df = df.dropna(thresh= ..., axis=0)
            #df = df.reset_index(drop = True)

            print("dataframe rows are not investigated yet")
    except:
        print("get_empty_values_info(): creation of dictionaries for NaN's of file {} is not possible, is df empty?".
             format(name))
        
    return dict_over25perc, dict_less26perc


#
# see: https://github.com/RianneSchouten/pymice/blob/master/pymice/exploration/mcar_tests.py
# small modifications have been done to fit our project
#
class McarTests():

    def __init__(self, data, name):
        self.data = data
        self.name = name
        

    def checks_input_mcar_tests(self, data):
        """ 
        Checks whether the input parameter of class McarTests is correct (private method of the McarTests class)
        
        Input:
            data - The input of McarTests specified as 'data'
        Output:
            bool - True if input is correct
        """

        if not isinstance(data, pd.DataFrame):
            print("Error: Data should be a Pandas DataFrame")
            return False

        if not any(data.dtypes.values == np.float):
            if not any(data.dtypes.values == np.int):
                print("Error: Dataset cannot contain other value types than floats and/or integers")
                return False

        if not data.isnull().values.any():
            print("Error: No NaN's in given data")
            return False

        return True

   
    def mcar_t_tests(self):
        """ 
        MCAR tests for each pair of variables
        
        Input:
            data - Pandas DataFrame
            An incomplete dataset with samples as index and variables as columns
        Output:
            mcar_matrix - Pandas DataFrame
            A square Pandas DataFrame containing True/False for each pair of variables
            True: Missingness in index variable is MCAR for column variable (note: > p value 0.05)
            False: Missingness in index variable is not MCAR for column variable
        """
        
        # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html

        if not self.checks_input_mcar_tests(self.data):
            raise Exception("Input not correct")

        dataset = self.data.copy()
        print("MCAR dataset '{}' shape: {}".format(self.name, dataset.shape))
        print("Creation of squared mcar matrix with shape 1 columns value for t-test.")
        vars = dataset.dtypes.index.values
        mcar_matrix = pd.DataFrame(data=np.zeros(shape=(dataset.shape[1], dataset.shape[1])),
                                   columns=vars, index=vars)
        
        # series types are created
        for var in vars:
            for tvar in vars:
                part_one = dataset.loc[dataset[var].isnull(), tvar].dropna()
                part_two = dataset.loc[~dataset[var].isnull(), tvar].dropna()
                #print("part_one: {}, part_one type: {}".format(part_one, type(part_one)))
                #print("part_two: {}, part_two type: {}".format(part_two, type(part_two)))
                # Note, March 2020: modified the params to array;
                # this project is a Python 3 implementation which is strongly typed compared to Python 2;
                # if not throws: TypeError: unsupported operand type(s) for /: 'str' and 'int'
                mcar_matrix.loc[var, tvar] = st.ttest_ind(part_one.array, part_two.array, equal_var=False).pvalue

        mcar_matrix = mcar_matrix[mcar_matrix.notnull()] > 0.05

        return mcar_matrix


def count(lst): 
    ''' Counts True values of a set or list by comprehension. '''
    return sum(bool(x) for x in lst)


def get_perc_True(df, name="", thresh=60):
    '''
    Returns dictionaries with matrix columns True information.
    
    Input:
        df - matrix of the dataframe created by MCAR t-test function before
        name - (string) name of the dataframe file
        thresh - threshold to build the 2 dictionaries, default value is 60% of True column values
    Output:
        dict_over60perc, dict_less61perc
    '''
    dict_over60perc = {}
    dict_less61perc = {}
    df_samples = df.shape[0]
    counter = 1
    
    list_clms = set(df.columns.to_list())
    print("\033[1mAmount of {} MCAR matrix columns included TRUE values:\033[0m".format(name))
    for col in list_clms:
        sum_True_col = count(df[col].array)
        
        perc_True_col = sum_True_col*100/df_samples
        if perc_True_col > thresh:
            dict_over60perc[col] = [sum_True_col, np.round(perc_True_col, decimals=2)]
        else:    # perc_True_col < 61
            dict_less61perc[col] = [sum_True_col, np.round(perc_True_col, decimals=2)]

        print("{}. '{}' includes: {}, means: \033[1m{:.4}%\033[0m".
              format(counter, col, sum_True_col, perc_True_col))
        counter += 1
        
    return dict_over60perc, dict_less61perc


# ANREDE_KZ
def add_others_ANREDE(df):
    '''
    Replaces the -1 value for unknown to 3 representing the category 'others'
    
    Input:
        df - dataframe including the feature 'ANREDE_KZ' to handle the replacement
    Output:
        df - the modified dataframe
    '''
    df['ANREDE_KZ'] = df['ANREDE_KZ'].replace(to_replace=-1, value=3)
    
    return df


# ALTER_HH
def calculate_ALTER_percentage(group, name=''):
    '''
    Calculates the percentage of age categories 0 up to 9 for the given groups build from dataset column 'ALTER_HH'.
    
    Input:
        group - the groupby result for column 'ALTER_HH'
        name - name of the dataset the calculation is based on
    Output:
        prints the calculated percentage and if one of the specific groups from 0 to 9 is not part of the calculation.
    '''
    limit = 10
    counter = 0.0
    sum_0_9 = 0
    while counter < limit:
        try:
            group_count = group.get_group(counter)['ALTER_HH'].count()
            #print("counter: {}; value: {}".format(counter, group_count))
            sum_0_9 += group_count
            counter += 1
        except KeyError:
            print("Key group '{}' does not exist in {} dataframe.".format(counter, name))
            counter += 1
            pass

    #print("sum_0_9: {}".format(sum_0_9))
    sum_ALTER = sum(group['ALTER_HH'].count())
    #print("whole sum: {}".format(sum_ALTER))
    perc_ALTER = sum_0_9 * 100 / sum_ALTER
    print("\033[1mResult for '{}' dataset:\033[0m".format(name))
    print("{:.4}% of the feature 'ALTER_HH' are unknown or unlikely to be customers.".format(perc_ALTER))


def helper_map_to_nan(df, attr, val):
    '''
    Helps to convert attribute values to NaN by calling that function internally.
    The Excel attribute information files includes not only integer values,
    string number sequences separated by ',' are available as well, their elements are converted to single numbers
    
    Input:
        df - dataframe to be changed
        attr - associated feature column of the dataframe that shall be modified
        val - associated value to be mapped to NaN
        
    Output:
        df - returns the modified dataframe
    '''
    if attr in df.columns:
        if isinstance(val, int):
            # val is integer
            df = map_to_nan(df, attr, val)
        elif isinstance(val, str):
            # val is string of few numbers separated by ',' 
            n_set = set(map(str.strip, val.split(',')))
            for n in n_set: 
                df = map_to_nan(df, attr, int(n))

    #print("\nThe new created dataframe is:")
    #print(df.head())
    #print('---')
    

def drop_rows(df, thresh):
    '''
    Deletes rows of given dataframe with number of missing values >= threshold.
    
    Input:
        df - the dataframe the rows shall be deleted from
        thresh - the treshold value for NaNs
    '''
    # by default axis=0, means rows are deleted
    df.dropna(thresh=thresh, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def scale_features(df):
    '''
    Scales the dataframe features with StandardScaler.
    '''
    df_clms = df.columns
    df_scaled = StandardScaler().fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled)
    df_scaled.columns = df_clms
    
    return df_scaled


###########################################
#
#  PCA part
#
###########################################

def pca_results(good_data, pca, dfname="file1", fgsize=(25,60)):
    '''
    Create a DataFrame of the PCA results
    Dataframe can be identified by its dfname (string)
    Includes dimension feature weights and explained variance
    Visualizes the PCA results (feature structure) of the last 15 dimensions
    '''

    # Dimension indexing
    dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = list(good_data.keys()))
    components.index = dimensions
    #print('components index:\n{}'.format(components.index))
    #print('------')
    if len(components.index) > 15:
        print('Last 15 index dimensions are from {} up to {}'.
              format(components.index[-15:-14][0], components.index[-1:][0]))
    print('------')
    
    # new dataframe with last 15 index dimensions
    df_last15_comp = components.loc[components.index[-15:]]
    #print('tail of df_last15_comp:\n{}'.format(df_last15_comp.tail()))

    # PCA explained variance (expl. var.)
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html#numpy.reshape
    # with n=1: result will be a 1-D array of that length
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions

    # Create a bar plot visualization and its colour map
    cmap = cm.get_cmap('gist_ncar')
    fig, ax = plt.subplots(figsize = fgsize)

    # Plot the feature weights as a function of the components   
    df_last15_comp.plot(ax=ax, kind='barh', colormap=cmap, fontsize=20, width=0.99)
    
    # right legend outside diagr
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", ncol=1, fontsize=20) 
    
    # Add a vertical line, here we set the style in the function call
    ax.axvline(x=0, ls='--', color='b')
    ax.set_xlabel("Feature Weights", fontsize=20)
    
    # Create dimension and explained variance ratio for each yticklabel
    last15_exp_var_ratios = pca.explained_variance_ratio_[-15:]
    #print('last15_exp_var_ratios:\n{}'.format(last15_exp_var_ratios))
    ylabels = []
    for i in range(0,15):
        label = df_last15_comp.index[i] + "\nExp. Variance: %.4f"%(last15_exp_var_ratios[i])
        ylabels.append(label)
        
    ax.set_yticklabels(ylabels) #df_last15_comp.index) #dimensions) #, rotation=45)  for bar chart
    ax.set_ylim(0, 15)
    ax.invert_yaxis()  # having first dimension on top

    # Display the explained variance ratios for bar diagram (not for barh: for that the y labels are used)
    #for i, ev in enumerate(pca.explained_variance_ratio_):
    #    ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Expl. Variance:\n     %.4f"%(ev))
    
    plt.show()
        
    # https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.savefig
    # save the diagram in a file
    filename = "./images/"+dfname+"_PCAdiagr.png"
    print("Store file: {}".format(filename))
    fig.savefig(filename, bbox_inches="tight")
    print("File stored ...")

    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis = 1)


def do_PCA(df, n_comp=15, name='', figsize=(25,60)):
    '''
    Creates and plots the PCA dataframe.
    
    Input:
        df - the dataframe we do the PCA (Principle Components Analysis) for.
        n_comp - the ordered components we want to get by priority (default value is 15);
                 using a value between 0<n_comp<1: the algorithm finds the minimum number of components
                 that fits to get this percentage of variance
        name - (string) name of the dataframe
        figsize - figure size the resulting plot shall have, default (25,60)
    Output:
        pca, pca_rslts
    '''
    # Apply PCA by fitting the modified data with the same number of dimensions as features
    print("Shape of the cleaned {} dataset is: {}".format(name, df.shape))
    if n_comp < 1:
        print("We calculate the dimensions for a variance of {}%".format(n_comp*100))
    else:
        print("We calculate {} dimensions.".format(n_comp))
    
    # check n_components ...
    # start with a default of 15, 
    # to identify which number is the best, use a value 0<val<1 as variance percentage, e.g. n_comp=0.85
    
    pca = PCA(n_components = n_comp, whiten = False, svd_solver = 'full')
    pca = pca.fit(df)

    # Generate PCA results plot
    pca_rslts = pca_results(df, pca, name, figsize)
    
    return pca, pca_rslts


#
# custering: Gaussian Mixture
#

# default starting point shall be n_components = 20 for cluster value
def create_clusterer(components, reduced_data):
    clusterer = GaussianMixture(n_components = components, random_state = 0).fit(reduced_data)
    return clusterer

# compare few silhouette coefficients
def create_silhouette_coeff(reduced_data, name, range_max=8):
    '''
    Creates a list of 13 silhouette coefficients to get the best cluster value for the given reduced dataframe.
    '''
    print("\033[1mSilhouette coefficients of the PCA reduced '{}' dataset:\033[0m".format(name))
    for comp in range(2, range_max):
        print("range {}".format(comp))
        clusterer = create_clusterer(comp, reduced_data)
        print('clusterer ready')

        # Predict the cluster for each data point
        preds = clusterer.predict(reduced_data)
        print('prediction ready')
        # Find the cluster centers
        centers = clusterer.means_
        print('cluster centers means are: {}'.format(centers))

        # Calculate the mean silhouette coefficient for the number of clusters chosen
        score = silhouette_score(X = reduced_data, labels = preds)
        print('GMM silhouette score of {} components: {}'.format(comp, score))



###########################################
#
# Visualisation part
#
###########################################

def plot_feature_bar(dct, title, xlabel, ylabel, labels, figsize=(8,6), title_fsize=None, label_size=None):
    '''
    Plots histograms out of values given by the dictionary param.
    
    Input:
        dct - dictionary including the needed x and y values for the bins 
        titel, xlabel, ylabel, labels, figsize, title_fsize, label_size - params to plot the labeled diagram
    '''
    # plot the dictionary keys as x params and the dictionary values as y params for the bar chart
    sorted_feature = sorted(dct.items(), key=lambda kv: kv[1])
    dict_sorted = collections.OrderedDict(sorted_feature)
    if labels in [None]:
        labels = list(dict_sorted.keys())
    values = list(dict_sorted.values())

    fig, ax = plt.subplots(figsize=figsize)
    plt.xticks(rotation=90)
    ax.bar(labels, values, alpha=0.8)
    if (title_fsize==None and label_size==None): 
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        fig.suptitle(title, fontweight='bold')   
    else: # only one case with PCA feature weights
        ax.set_ylabel(ylabel, fontsize=label_size)
        ax.set_xlabel(xlabel, fontsize=label_size)
        fig.suptitle(title, fontweight='bold', fontsize=title_fsize)
        filepath = "./images/az_pca_rslt_featWeightsSum-90Var_fig.png"
        plt.savefig(filepath)
        print("File stored ...")
        

def plot_NaN_distribution(feat_labels, values_set1, values_set2, label_set1, label_set2, figsize=(10, 8.5), xmax=6.5, barwidth=0.35):
    '''
    Plots the NaN distribution being higher 35% for 2 dataset values
    
    Input:
        feat_labels - feature list to be shown by bars
        values_set1 - list of NaN percentage values of dataset 1 dictionary (default: must be higher 35%)
        values_set2 - list of NaN percentage values of dataset 2 dictionary (default: must be higher 35%)
        label_set1 - general string name of dataset 1
        label_set2 - general string name of dataset 2
        figsize - (optional) default value (10, 6)
        xmax - needed to draw the 35% horizontal line
        barwidth - width of the bars, depends on the amount of them
    Output:
        the plotted bar diagram
    '''

    x = np.arange(len(feat_labels))  # the label locations
    width = barwidth  # the width of the bars, 0.35 as default

    fig, ax = plt.subplots(figsize=figsize)
    rects1 = ax.bar(x - width/2, values_set1, width, label=label_set1, alpha=0.9)
    rects2 = ax.bar(x + width/2, values_set2, width, label=label_set2, alpha=0.9)

    # Add some text for labels, title and custom axis tick labels, etc.
    ax.set_title('Empty Value Feature Distribution >35% of both Datasets', size=14)
    ax.set_xticks(x)
    plt.xticks(rotation=90)
    ax.set_xticklabels(feat_labels)
    extratick = [35]
    ax.set_yticks(list(ax.get_yticks()) + extratick)
    ax.legend(loc='best', frameon=False)
    ax.hlines(y=35,  xmin=-0.5, xmax=xmax, linewidth=2, color='b', linestyle='-')
    ax.set_ylabel('% percentage')
    ax.set_xlabel('features')

    def autolabel(rects, heights=None):
        '''
        Attaches a text label above each bar in *rects*, displaying its height.
        '''
        if heights is None:  # bar chart
            for rect in rects:
                height = rect.get_height()          
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=12)
        else:  # histogram
            i = 0
            for rect in rects:
                ax.annotate('{}'.format(heights[i]),
                            xy=(rect, heights[i]),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='left', va='bottom')
                i += 1         

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout();


def plot_histogram(data, xlabel, dataset_name):
    '''
    Plots the density distribution of a feature.
    
    Input:
        data - the dataframe column data to be visualised
        xlabel - feature string name
        dataset_name - string name of the dataset the column comes from
    Output:
        density diagram
        
    Note:
    use distplot (figure-level function) or histplot (axes-level function), axes level will be removed   
    '''
    plt.subplots(figsize=(6, 4))
    grafic = sns.distplot(data, rug=True)
    grafic.tick_params(labelsize=10)
    grafic.set_ylabel('Frequency', fontsize=11)
    grafic.set_xlabel(xlabel, fontsize=11)
    title = ('Density Distribution of {} Dataset'.format(dataset_name))
    plt.title(title, fontsize=12, fontweight='bold');


# from: https://github.com/ResidentMario/missingno
# with small modifications: changed figsize, save chart to file, msno func calls, add df name and ';' for axes
def heatmap(df, dfname="", inline=False,
            filter=None, n=0, p=0, sort=None,
            figsize=(90, 75), fontsize=16, labels=True, 
            cmap='RdBu', vmin=-1, vmax=1, cbar=True, ax=None
            ):
    """
    Presents a `seaborn` heatmap visualization of nullity correlation in the given DataFrame.
    
    Note that this visualization has no special support for large datasets. For those, try the dendrogram instead.
    :param df: The DataFrame whose completeness is being heatmapped.
    :param filter: The filter to apply to the heatmap. Should be one of "top", "bottom", or None (default). See
    `nullity_filter()` for more information.
    :param n: The cap on the number of columns to include in the filtered DataFrame. See  `nullity_filter()` for
    more information.
    :param p: The cap on the percentage fill of the columns in the filtered DataFrame. See  `nullity_filter()` for
    more information.
    :param sort: The column sort order to apply. Can be "ascending", "descending", or None.
    :param figsize: The size of the figure to display. This is a `matplotlib` parameter which defaults to (20, 12).
    :param fontsize: The figure's font size.
    :param labels: Whether or not to label each matrix entry with its correlation (default is True).
    :param cmap: What `matplotlib` colormap to use. Defaults to `RdBu`.
    :param vmin: The normalized colormap threshold. Defaults to -1, e.g. the bottom of the color scale.
    :param vmax: The normalized colormap threshold. Defaults to 1, e.g. the bottom of the color scale.
    :param inline: Whether or not the figure is inline. If it's not then instead of getting plotted, this method will
    return its figure.
    :return: If `inline` is False, the underlying `matplotlib.figure` object. Else, nothing.
    """
    # Apply filters and sorts, set up the figure.
    df = msno.utils.nullity_filter(df, filter=filter, n=n, p=p)
    df = msno.utils.nullity_sort(df, sort=sort, axis='rows')

    if ax is None:
        plt.figure(figsize=figsize)
        ax0 = plt.gca()
    else:
        ax0 = ax

    # Remove completely filled or completely empty variables.
    df = df.iloc[:,[i for i, n in enumerate(np.var(df.isnull(), axis='rows')) if n > 0]]

    # Create and mask the correlation matrix. Construct the base heatmap.
    corr_mat = df.isnull().corr()
    mask = np.zeros_like(corr_mat)
    mask[np.triu_indices_from(mask)] = True

    if labels:
        sns.heatmap(corr_mat, mask=mask, cmap=cmap, ax=ax0, cbar=cbar,
                    annot=True, annot_kws={'size': fontsize - 2},
                    vmin=vmin, vmax=vmax);
    else:
        sns.heatmap(corr_mat, mask=mask, cmap=cmap, ax=ax0, cbar=cbar,
                    vmin=vmin, vmax=vmax);

    # Apply visual corrections and modifications.
    ax0.xaxis.tick_bottom()
    ax0.set_xticklabels(ax0.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=fontsize)
    ax0.set_yticklabels(ax0.yaxis.get_majorticklabels(), fontsize=fontsize, rotation=0)
    ax0.set_yticklabels(ax0.yaxis.get_majorticklabels(), rotation=0, fontsize=fontsize)
    ax0.xaxis.set_ticks_position('none')
    ax0.yaxis.set_ticks_position('none')
    ax0.patch.set_visible(False);

    for text in ax0.texts:
        t = float(text.get_text())
        if 0.95 <= t < 1:
            text.set_text('<1')
        elif -1 < t <= -0.95:
            text.set_text('>-1')
        elif t == 1:
            text.set_text('1')
        elif t == -1:
            text.set_text('-1')
        elif -0.05 < t < 0.05:
            text.set_text('')
        else:
            text.set_text(round(t, 1))
    
    filepath = "./images/" + dfname + "_mdfd_nullcorr_figure.png"
    if inline:
        warnings.warn(
            "The 'inline' argument has been deprecated, and will be removed in a future version "
            "of missingno."
        )
        plt.savefig(filepath)
        print("File stored ...")
        plt.show()
    else:
        ax0.figure.savefig(filepath)
        print("File stored ...")
        return ax0;

