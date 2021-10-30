[//]: # (Image References)

[image1]: ./images/ClassificationModelsEvalResultSmall.PNG "Evaluation of Classification models:"
[image2]: ./images/kaggleCompetition_positionUdacityDataScience-BertelsmannArvato-141_2021-08-15.JPG "Kaggle ranking:"

# Customer Segmentation Report for Arvato Financial Services

<b>Note:</b> Unfortunately the notebook files and all their associated .html files are each bigger ones (>25 MB) and therefore could not be stored in this repository!

## Project Overview
With this <b>Udacity Data Science Capstone Project</b>, we are investigating the process chain of a <b>customer relationship</b> of the German, internationally active <b>Bertelsmann Group and its service provider arvato</b>. This German mail-order sales company is interested in identifying segments of the general population to target with their marketing in order to grow. This is their business goal. With this project, we have the role to provide necessary information as a basis for decision-making. 

So, we analyse demographics information of Bertelsmann customer data and compare it against such information from the general population. The underlying data are under restrictions from Bertelsmann - arvato and therefore not stored in this repository.

From <b>Data Science</b> point of view, we start with the ETL (Extract, Transform, Load) pipeline including EDA (Exploratory Data Analysis) activities, afterwards we take care of the customer segmentation to build appropriate clusters as the unsupervised learning part, followed by the ML (Machine Learning) pipeline for the investigated classification prediction models as the supervised learning part. The project ends up with a Kaggle competition to get a ranking about our best found classification model and its predictions compared with fellow students.

### Delivered files
There are four data files associated with this project:

* <i>Udacity_AZDIAS_052018.csv:</i> Demographics data for the general population of Germany; 891 211 person (rows) x 366 features (columns)
* <i>Udacity_CUSTOMERS_052018.csv:</i> Demographics data for customers of a mail-order company; 191 652 person (rows) x 369 features (columns)
* <i>Udacity_MAILOUT_052018_TRAIN.csv:</i> Demographics data for individuals who were targets of a marketing campaign; 42 982 person (rows) x 367 (columns)
* <i>Udacity_MAILOUT_052018_TEST.csv:</i> Demographics data for individuals who were targets of a marketing campaign; 42 833 person (rows) x 366 (columns)

And 2 MS Excel files for feature explanation:

* <i>DIAS Information Levels - Attributes 2017.xlsx:</i> Top-level list of attributes and descriptions, organised by informational category
* <i>DIAS Attributes - Values 2017.xlsx:</i> Detailed mapping of data values for most of the features in alphabetical order

But there are German column labels which are not explained in the mentioned Excel files. Here are some additional attribute explanations:

* ALTER_KIND1 - age of child no. 1
* ALTER_KIND2 - age of child no. 2
* ALTER_KIND3 - age of child no. 3
* ALTER_KIND4 - age of child no. 4
* ALTERKATEGORIE_FEIN - age categories ('fein' in this case I guess means a more specific classification)
* ANZ_KINDER - no. of children
* ANZ_STATISTISCHE_HAUSHALTE - in Germany a census data is given about statistical households, here its number is counted
* ARBEIT - work
* D19_KONSUMTYP_MAX - what kind of consumption type you are
* D19_LETZTER_KAUF_BRANCHE - the last date you bought something and info about category/segment
* D19_VERSI_DATUM - insurance date
* D19_VERSI_OFFLINE_DATUM - insurance date offline
* D19_VERSI_ONLINE_DATUM - insurance date online
* D19_VERSI_ONLINE_QUOTE_12 - insurance online quote in the last 12 month
* DSL_FLAG - what kind of DSL are you using/ are you in
* EINGEFUEGT_AM - inserted on
* EINGEZOGEN_HH_JAHR - moved to household in mentioned year
* FIRMENDICHTE - company density
* KOMBIALTER - 'Kombi' means combination and 'Alter' is age
* STRUKTURTYP - what kind of structural type you are
* UMFELD_ALT - surrounding/neighbourhood is old
* UMFELD_JUNG - surrounding/neighbourhood is young
* VERDICHTUNGSRAUM - urban agglomeration
* VERS_TYP - what kind of insurance type you are

### Short summary
Regarding the <b>workflow</b> and the most prominent tasks and results, you can read my following <b>medium</b> [blog post](https://medium.com/@ilona.brinkmeier/customer-segmentation-report-for-arvato-financial-solutions-167cba1545bd).

For this <b>coding</b>, main project parts are:

1. The <i>unsupervised part</i> of customer segmentation creating clusters.<br>
  We have identified 3 main groups of Bertelsmann customers with unique properties:
  
  * Customers having a higher education with <i>academic titles</i> (ca. 23%)
  * Customers which are <i>ecological oriented and having children</i> (ca. 16%)
  * Customers being an <i>‘average’ family</i> (ca. 13.5%)
  
2. The <i>supervised learning</i> model to deliver a binary classification algorithm predicting if a person is being a Bertelsmann — arvato campaign candidate or not.<br>
  Our evaluation results of the metrics Accuracy and ROC AUC of the investigated classification models is:
  ![Evaluation of Classification models:][image1]
3. The final [<i>Kaggle competition</i>](https://www.kaggle.com/c/udacity-arvato-identify-customers) where the scoring with other fellow students happens.<br>
  My PoC ranking for the first submitted best model try is:
  ![Kaggle ranking:][image2]

And about the <b>notebooks</b>:<br>
The first notebook - <i>Arvato_Project_Workbook-PoC-Part1_V01</i> - includes the ETL and the unsupervised learning topics, the supervised learning and the Kaggle competition ranking part of the project are implemented in a second Jupyter notebook - <i>Arvato_Project_Workbook-PoC-Part2-3_V01</i>.

### Additional comment
In general, I would separate the tasks having few Python files, but implementing such approach it is necessary to store the modified datasets as a new SQL database each. This was not possible with my own equipment, it crashed with memory errors, therefore only 2 notebook files are implemented. As a consequence, the dereferencing and deletion of variables not needed anymore is implemented often to improve garbage collection.

Furthermore, regarding engineering principles, this project is implemented as a PoC (Proof-of-Concept) only. Means, no primary SW architecture is designed and the coding is not separated and grouped to Python classes having included domain specific tasks. No packages are created to be able to deploy a real project result including a production application to handle new data records. All this would be a future toDo.

## Technical
All coding has been implemented under virtual environment conditions with the libraries mentioned in the requirements.txt file. For Windows, having conda installed, the installation can be done e.g. by:
- __Windows__
  ```
	conda create --name arvato-project python=3.7
	activate arvato-project
	pip install -r requirements/requirements.txt
  ```
  and for a Jupyter notebook
  ```
  python -m ipykernel install --user --name arvato-project --display-name "arvato-project"
  ```

## License
This project coding is released under the [MIT](https://github.com/IloBe/Customer-Segmentation-and-Classification-Master/blob/master/LICENSE) license.
