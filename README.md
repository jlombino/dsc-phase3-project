<h1>SyriaTel Customer Churn Project</h1>

Author: Jason Lombino

For more information on this project please see my <a href= .slides.pdf>Presentation</a> or <a href= ./Final_Notebook.ipynb>Jupyter Notebook</a>
<br>

<h1>Business Problem</h1>

Reducing customer churn is an important part of running a successful business. It is more expensive to acquire new customers through advertising and promotions than it is to keep existing customers. In addition, it is often the customers who are paying the most who are the fastest to switch to a competitor for better pricing. Telecomunications company SyriaTel would like to focus on retaining customers by offering discounted rates to customers who are likely to leave soon. In order to do this, SyriaTel needs a model to predict which customers are likely to churn. 

Provided with data on customers' accounts, the model should be helpful in answering the following questions:


<ol>
<li>Will a given customer leave SyriaTel soon?
<li>Which account features best predict whether a customer will soon churn?
</ol>

SyriaTel should be able to use this model to target all customers who will churn with discounted rates while avoiding discounting rates for customers who will not. My primary metric for the model will be <b>Recall Score</b> because it is most important that the model correctly identifies as many churning customers as possible. I will then focus on Precision Score to avoid giving out unnecessary discounts to customers who will not churn.

<h1>Data</h1>

The following <a href=https://www.kaggle.com/becksddf/churn-in-telecoms-dataset>dataset</a> was provided by SyriaTel for modeling. It contains information on the account usage and history of 3300 SyriaTel customers in the United States. The target column {churn} shows whether a given customer left SyriaTel during an unspecified time frame. The dataset can be found in this repositiory at <a href=./data/s_tel.csv>./data/s_tel.csv</a>.

The features present in the dataset include:
<ul>      
<li>churn (Target)
<li>account length
<li>total day minutes     
<li>total day calls       
<li>total day charge      
<li>total intl minutes
<li>total intl calls
<li>total intl charge
<li>customer service calls
</ul> 

<h2>Class Imbalance</h2>

One major consideration is that there are six times more non-churners than churners in the dataset. This can be a problem for many models and will need to be addressed using a method such as weighting the data points by class.

<img src=./images/classes.png><br>

<h1>Exploratory Data Analysis</h1>

<h2>Feature Distrobutions</h2>

I will begin by plotting the distrobution of each of the predictor columns.

<img src=./images/dists.png><br>

<h2>Correlation Among Features</h2>

I will also take a look at the correlation coefficients between each of the predictor columns. Here, it looks like some of the columns are perfectly correlated. It makes sense that total charge would be an integer multiple of minutes, so to avoid issues with multicolinearity I will later drop all of the minutes features and keep the charge features.

<img src=./images/corr.png><br>

<h2>Useful Features</h2>

Now, I will take a look at all of the remaining features separately. I will divide each of the features into churned and not-churned groups and compare the groups using a statistical test. The goal is to determine whether a customer will churn, so I will be looking for a significant difference between the groups for each feature. While I won't be dropping and data yet, this should help me get an idea of which features will be the most useful predictors of churn.

<h3>Total Day Charge</h3> 

Total Day Charge is an example of a feature I found to be promising. There is a big difference in the distrobution of total day charges for the churn and non-churn groups.

<img src=./images/tdc.png><br>

<h3>Total Day Calls</h3> 

Total Day Calls is an example of a feature I thought would not be useful. There is not really a difference in the distrobution of total day calls for the churn and non-churn groups.

<img src=./images/tdcall.png><br>

<h2>Summary</h2>

I have grouped the features as followed based on the plots and statistical tests. 

Target:
<ul><li>churn</ul>

Dropping Duplicates:
<ul><li>total day minutes
<li>total eve minutes
<li>total night minutes
<li>total intl minutes</ul>

Less Useful / Consider Dropping:
<ul><li>state
<li>account length
<li>total day calls
<li>total eve calls
<li>total night calls</ul>

Useful:
<ul><li>international plan
<li>voice mail plan
<li>number vmail messages
<li>total day charge
<li>total eve charge
<li>total night charge
<li>total intl calls
<li>total intl charge
<li>customer service calls</ul>

<h1>Feature Engineering</h1>

I created new features which may be useful for predicting whether a customer will churn.

<h3>Total Charge</h3> 

Total Charge is the first feature I created by summing total day, eve, night, and intl charge features.

<img src=./images/tch.png><br>

<h3>Total Calls</h3> 

Total Day Calls is the second feature I created by summing the total day, eve, night, and intl call features.

<img src=./images/tca.png><br>

<h1>Preliminary Modeling</h1>

I began by testing a variety of models using their default settings. I attempted to correct for the ~6:1 class imbalence using both class weights (CW) and SMOTE where appropriate. This helped me determine which models I should focus on optimizing later.

I used the following models:
<ol>
<li>Logistic Regression
<li>Decision Tree
<li>K Neighbors
<li>Extra Trees
<li>Random Forest
<li>AdaBoost
<li>XGBoost
</ol>

<h2>Results</h2>
<ul>
<li>Logistic Regression - CW: Recall Score: 0.752
<li>Logistic Regression - SMOTE: Recall Score: 0.749
<li>Decision Tree - CW: Recall Score: 0.860
<li>Decision Tree - SMOTE: Recall Score: 0.865
<li>KNN - Vanilla: Recall Score: 0.370
<li>KNN - SMOTE: Recall Score: 0.676
<li>Extra Trees - CW: Recall Score: 0.591
<li>Extra Trees - SMOTE: Recall Score: 0.739
<li>Random Forest - CW: Recall Score: 0.790
<li>Random Forest - SMOTE: Recall Score: 0.860
<li>AdaBoost - Vanilla: Recall Score: 0.549
<li>AdaBoost - SMOTE: Recall Score: 0.782
<li>XGBoost - Vanilla: Recall Score: 0.863
<li>XGBoost - SMOTE: Recall Score: 0.868
</ul>

Once again, the primary metric I am using to evaluate models is <b>Recall Score</b>.

Three models stand out:

<ul>
<li>Decision Tree
<li>Random Forest
<li>XGBoost
</ul>

I skipped optimizing the decision tree in favor of the random forest because the random forest has more parameters I could tune. Both the random forest and XGBoost models performed better when using SMOTE than class weights so I continued using SMOTE when optimizing the models further.

<h1>Model Tuning</h1>

The main problem with both models is that they seem to be overfitting. This is evidenced by the scores on training being significantly higher than the scores on cross validation. I will attempt to address this by reducing the number of features my model has to train on and using a grid search to find the optimal hyperparemeters for the model.

I am selecting the features to drop based on the feature importances from the original random forest and my exploratory data analysis. I am then refitting a random forest on the new feature set and comparing the resutls to the original random forest. Feature selection does appear to improve the predictions of the model, but does not seem to be doing a lot about the overfitting problem.

Here, I am using the new feature set and looping over a variety of hyperparameters of the random forest model. The main goal is to improve the model's predictions by reducing overfitting. This was relatively successful as the model's cross validation scores improved and got closer to the training data scores indicating the model is overfitting less.

This is the best I was able to get a random forest to perform.

<h1>Final Model Evauluation</h1>

Here is a comparison between the optimized random forest and XGBoost models. I selected the XGBoost model as my final model because it tends to overfit the data less than the random forest and the cross validation scores are nearly identical.

['Accuracy: 0.972', 'Precision: 0.964', 'Recall: 0.835', 'F1: 0.895']

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

This model has a built in feature importances method. Not all models have this method and the ones that do are not always calculated based on the same method. Therefore, the SHAP scores below are preferred.

<h3>Using SHAP (SHapley Additive exPlanations) </h3>

This shows the feature importances for the XGBoost model as well as how each feature contributed to a prediction on average.

<img src=./images/shap_imp.png><br>

This plot shows the feature contributions for each feature for every prediction on the train set. Color is the value of the feature and position (left or right) is the contribution of that feature for a given point.

<img src=./images/shap_swarm.png><br>

This plot shows the feature contributions for one model prediction from the train set.

<img src=./images/shap_pred.png><br>

<h1>Conclusion</h1>

I was successful in creating a model that SyriaTel can use to predict whether a customer will churn soon. The best model was XGBoost with <b>83.5% Recall</b> and 96.4% Precision on the test set.

The most important features for predicting churn are:
<ol><li>Total Charge
<li>Customer Service Calls
<li>International Plan
<li>Total International Calls
<li>Total International Charge
</ol>

SyriaTel can use this model to offer discounts to customers who are likely to churn soon while avoiding offering unnecessary discounts to customers who are unlikely to do so.

 # Repository Information
```
├── README.md                        <- The top-level README you are currently reading
├── Description.md                   <- Project description and requirements provided by upstream
├── Final_Notebook.ipynb             <- Jupyter notebook with my full analysis
├── Final_Notebook.pdf               <- PDF version of project Jupyter notebook
├── Pres_Graphs.ipynb                <- Jupyter notebook used to make graphs for presentation
├── slides.pdf                       <- PDF version of project presentation
├── data                             <- Project data provided by upstream
└── images                           <- Graphs generated from code
```