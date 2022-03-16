<h1>SyriaTel Customer Churn Project</h1>

Author: Jason Lombino

For more information on this project please see my <a href= .slides.pdf>Presentation</a> or <a href= ./Final_Notebook.ipynb>Jupyter Notebook.</a>
<br>

<h1>Business Problem</h1>

Reducing customer churn is an important part of running a successful business. It is more expensive to acquire new customers through advertising and promotions than it is to keep existing customers. In addition, it is often the customers who are paying the most who are the fastest to switch to a competitor for better pricing. Telecomunications company SyriaTel would like to focus on retaining customers by offering discounted rates to customers who are likely to leave soon. In order to do this, SyriaTel needs a model to predict which customers are likely to churn. 

Provided with data on customers' accounts, the model should be helpful in answering the following questions:


<ol>
<li>Will a given customer leave SyriaTel soon?
<li>Which account features best predict whether a customer will soon churn?
</ol>

SyriaTel should be able to use this model to target all customers who will churn with discounted rates while avoiding discounting rates for customers who will not. The primary metric for the model will be <b>Recall Score</b> because it is most important that the model correctly identifies as many churning customers as possible. Precision Score will be a secondary focus to avoid giving out unnecessary discounts to customers who will not churn.

<h1>Data</h1>

The following <a href=https://www.kaggle.com/becksddf/churn-in-telecoms-dataset>dataset</a> was provided by SyriaTel for modeling. It contains information on the account usage and history of 3300 SyriaTel customers in the United States. The target column, churn shows whether a given customer left SyriaTel during an unspecified time frame. The dataset can be found in this repositiory at <a href=./data/s_tel.csv>./data/s_tel.csv</a>.

Some of the features present in the data set are:
<ul>      
<li>Churn (Target)
<li>Account length
<li>Total day charge      
<li>Total international charge
<li>Customer service calls
</ul> 

<h2>Class Imbalance</h2>

One major consideration is that there are six times more non-churned than churned customers in the dataset. This can be a problem for many models and will need to be addressed using a method such as weighting the data points by class.

<img src=./images/classes.png><br>

<h1>Exploratory Data Analysis</h1>

Before any models are run, it is always useful to take a look at the features of the dataset.

<h2>Feature Distributions</h2>

The following plot shows how each of the features in the dataset are distributed. Everything except voicemail messages, customer service calls, and international calls look close to normally distributed.

<img src=./images/dists.png><br>

<h2>Correlation Among Features</h2>

The following plot shows the correlations between each of the predictor columns. It looks like some of the columns are perfectly correlated - total charges are just imteger multiples of minutes. To avoid issues with multicolinearity all of the minutes features should be dropped in favor of the charge features.

Duplicate features to be dropped:
<ul><li>Total day minutes
<li>Total eve minutes
<li>Total night minutes
<li>Total intl minutes</ul>

<img src=./images/corr.png><br>

<h1>Selecting Useful Features</h1>

The difference between the churned and non-churned cusomters can be considered for each feature separately. Comparisons between the churned and non-churned groups can then be made to determine whether a given feature will be useful for separating the two groups. Features that can separate the groups well are more likely to be useful to a model predicting which group a customer belongs to.

<h2>Total Day Charge</h2> 

Total Day Charge is an example of a feature that separates the churned and non-churned groups well as there is a significant difference in the median value for each group.

<img src=./images/tdc.png><br>

<h2>Total Day Calls</h2> 

Total Day Calls is an example of a feature that does not separate the churned and non-churned groups well. There is no significant difference in the median value for each group.

<img src=./images/tdcall.png><br>

Features that separate the data well include:

<ul>
<li>International plan
<li>Voice mail plan
<li>Number vmail messages
<li>Total day charge
<li>Total eve charge
<li>Total night charge
<li>Total intl calls
<li>Total intl charge
<li>Customer service calls
</ul>

Features that do not separate the data well include:
<ul><li>State
<li>Account length
<li>Total day calls
<li>Total eve calls
<li>Total night calls</ul>

<h1>Feature Engineering</h1>

Creating new features based on existing features may be useful for separating the customers into churned and non-churned groups.

<h2>Total Charge</h2> 

Total Charge is the sum of the total day, eve, night, and international charge features and appears to separate the customers into groups well.

<img src=./images/tch.png><br>

<h1>Preliminary Modeling</h1>

The primary metric used to evaluate models for this project is <b>Recall Score</b>. Precision score is also used as a secondary metric. Class weights and SMOTE were used to correct for the ~6:1 class imbalence where appropriate.

The following preliminary models were used to attempt to classify customers into churned and non-churned groups:
<ol>
<li>Logistic Regression
<li>Decision Tree
<li>K Neighbors
<li>Extra Trees
<li>Random Forest
<li>AdaBoost
<li>XGBoost
</ol>

The three top performing models with default parameters were:

<ul>
<li>Decision Tree
<li>Random Forest
<li>XGBoost
</ul>

The random forest and XGBoost models were selected for optimization because they had the most parameters available to tune for performance. Both the random forest and XGBoost models performed better using SMOTE than class weights in cross validation.

<h1>Model Tuning</h1>

The main problem with both models is that they seem to be overfitting. This is evidenced by the scores on training being significantly higher than the scores on cross validation. This can be addressed by reducing the number of features each model has to train on and using a grid search to find the optimal hyperparameters for each model.

Features to drop were selected based on the exploratory data analysis and feature importances in the original models. Feature selection alone did appear to improve the predictions of the model, but did not solve the problem of overfitting to the training data.

Grid search with cross validation was used to loop over a veriety of hyperparameters and optimize each model. Feature selection and grid search together significantly improved the models' overfitting problems.


<h1>Final Model Evauluation</h1>

Here is a comparison between the optimized random forest and XGBoost models. The XGBoost model was selected as the final model because it tends to overfit the data less than the random forest and the cross validation scores are nearly identical.

XGBoost test set scores:
<ul>
<li>Accuracy: 0.972
<li>Precision: 0.964
<li>Recall: 0.835
<li>F1: 0.895
</ul>

Here is the confusion matrix for the optimized XGBoost model on the test set.

<img src=./images/confusion.png><br>

<h1>Feature Importances</h1>

The most important features for predicting churn are:
<ol><li>Total Charge
<li>Customer Service Calls
<li>International Plan
<li>Total International Calls
<li>Total International Charge
</ol>

<h3>From the XGBoost Model</h3>

SHAP values are preferred to the built-in feature importances method because they can be applied to any model.

<h3>Using SHAP (SHapley Additive exPlanations)</h3>

This shows how much each feature contributed to a prediction for the XGBoost model on average.

<img src=./images/shap_imp.png><br>

This plot shows the contribution of each feature for every prediction on the train set. Color is the value of the feature and position is the contribution of that feature for a given point.

<img src=./images/shap_swarm.png><br>

This plot shows the feature contributions for one model prediction from the train set.

<img src=./images/shap_pred.png><br>

<h1>Conclusion</h1>

The final model can be used by SyriaTel to predict whether a customer will churn soon with <b>83.5% Recall</b> and 96.4% Precision. 

The most important features for predicting churn are:
<ol><li>Total Charge
<li>Customer Service Calls
<li>International Plan
<li>Total International Calls
<li>Total International Charge
</ol>

SyriaTel can use this model to offer discounts to customers who are likely to churn soon while avoiding offering unnecessary discounts to customers who are unlikely to do so.

 <h1>Repository Information</h1>

```
├── README.md                        <- The top-level README you are currently reading
├── Description.md                   <- Project description and requirements provided by upstream
├── Final_Notebook.ipynb             <- Jupyter notebook with my full analysis
├── Final_Notebook.pdf               <- PDF version of project Jupyter notebook
├── slides.pdf                       <- PDF version of project presentation
├── data                             <- Project data provided by upstream
└── images                           <- Graphs generated from code
```