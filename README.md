[//]: # (Image References)

[image1]: ./images/ClassificationModelsEvalResultSmall.PNG "Evaluation of Classification models:"
[image2]: ./images/kaggleCompetition_positionUdacityDataScience-BertelsmannArvato-141_2021-08-15.JPG "Kaggle ranking:"

# Customer Segmentation Report for Arvato Financial Services

## Project Overview
With this <b>Udacity Data Science Capstone Project</b>, we are investigating the process chain of a <b>customer relationship</b> of the German, internationally active <b>Bertelsmann Group and its service provider arvato</b>. This German mail-order sales company is interested in identifying segments of the general population to target with their marketing in order to grow. This is their business goal. With this project, we have the role to provide necessary information as a basis for decision-making. 

So, we analyse demographics information of Bertelsmann customer data and compare it against such information from the general population. The underlying data are under restrictions from Bertelsmann - arvato and therefore not stored in this repository.

From <b>Data Science</b> point of view, we start with the ETL (Extract, Transform, Load) pipeline including EDA (Exploratory Data Analysis) activities, afterwards we take care of the customer segmentation to build appropriate clusters as the unsupervised learning part, followed by the ML (Machine Learning) pipeline for the investigated classification prediction models as the supervised learning part. The project ends up with a Kaggle competition to get a ranking about our best found classification model and its predictions compared with fellow students.

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
