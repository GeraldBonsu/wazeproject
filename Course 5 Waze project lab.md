# **Waze Project**
**Course 5 - Regression analysis: Simplify complex data relationships**

Your team is more than halfway through their user churn project. Earlier, you completed a project proposal, used Python to explore and analyze Waze’s user data, created data visualizations, and conducted a hypothesis test. Now, leadership wants your team to build a regression model to predict user churn based on a variety of variables.

You check your inbox and discover a new email from Ursula Sayo, Waze's Operations Manager. Ursula asks your team about the details of the regression model. You also notice two follow-up emails from your supervisor, May Santner. The first email is a response to Ursula, and says that the team will build a binomial logistic regression model. In her second email, May asks you to help build the model and prepare an executive summary to share your results.

A notebook was structured and prepared to help you in this project. Please complete the following questions and prepare an executive summary.

# **Course 5 End-of-course project: Regression modeling**

In this activity, you will build a binomial logistic regression model. As you have learned, logistic regression helps you estimate the probability of an outcome. For data science professionals, this is a useful skill because it allows you to consider more than one variable against the variable you're measuring against. This opens the door for much more thorough and flexible analysis to be completed.
<br/>

**The purpose** of this project is to demostrate knowledge of exploratory data analysis (EDA) and a binomial logistic regression model.

**The goal** is to build a binomial logistic regression model and evaluate the model's performance.
<br/>

*This activity has three parts:*

**Part 1:** EDA & Checking Model Assumptions
* What are some purposes of EDA before constructing a binomial logistic regression model?

**Part 2:** Model Building and Evaluation
* What resources do you find yourself using as you complete this stage?

**Part 3:** Interpreting Model Results

* What key insights emerged from your model(s)?

* What business recommendations do you propose based on the models built?

<br/>

Follow the instructions and answer the question below to complete the activity. Then, you will complete an executive summary using the questions listed on the PACE Strategy Document.

Be sure to complete this activity before moving on. The next course item will provide you with a completed exemplar to compare to your own work.

# **Build a regression model**

<img src="images/Pace.png" width="100" height="100" align=left>

# **PACE stages**


Throughout these project notebooks, you'll see references to the problem-solving framework PACE. The following notebook components are labeled with the respective PACE stage: Plan, Analyze, Construct, and Execute.

<img src="images/Plan.png" width="100" height="100" align=left>


## **PACE: Plan**
Consider the questions in your PACE Strategy Document to reflect on the Plan stage.

### **Task 1. Imports and data loading**
Import the data and packages that you've learned are needed for building logistic regression models.


```python
# Packages for numerics + dataframes
import pandas as pd
import numpy as np

# Packages for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Packages for Logistic Regression & Confusion Matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, \
recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression

```

Import the dataset.

**Note:** As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.


```python
# Load the dataset by running this cell

df = pd.read_csv('waze_dataset.csv')
```

<img src="images/Analyze.png" width="100" height="100" align=left>

## **PACE: Analyze**

Consider the questions in your PACE Strategy Document to reflect on the Analyze stage.

In this stage, consider the following question:

* What are some purposes of EDA before constructing a binomial logistic regression model?

When I run logistic regression models, I know that outliers and extreme data values can have a big impact. After visualizing the data, my plan is to handle outliers by either dropping certain rows, replacing extreme values with averages, or removing values that fall more than three standard deviations away from the mean.

As part of my EDA, I also check for missing data so I can decide whether to exclude it or fill it in using methods like substituting with the mean, median, or another suitable approach.

I also like to engineer new variables where it makes sense, such as multiplying features together or creating ratios. For example, in this dataset I could create a drives_sessions_ratio by dividing drives by sessions.

### **Task 2a. Explore data with EDA**

Analyze and discover data, looking for correlations, missing data, potential outliers, and/or duplicates.



Start with `.shape` and `info()`.


```python
print(df.shape)

df.info()
```

    (14999, 13)
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 14999 entries, 0 to 14998
    Data columns (total 13 columns):
     #   Column                   Non-Null Count  Dtype  
    ---  ------                   --------------  -----  
     0   ID                       14999 non-null  int64  
     1   label                    14299 non-null  object 
     2   sessions                 14999 non-null  int64  
     3   drives                   14999 non-null  int64  
     4   total_sessions           14999 non-null  float64
     5   n_days_after_onboarding  14999 non-null  int64  
     6   total_navigations_fav1   14999 non-null  int64  
     7   total_navigations_fav2   14999 non-null  int64  
     8   driven_km_drives         14999 non-null  float64
     9   duration_minutes_drives  14999 non-null  float64
     10  activity_days            14999 non-null  int64  
     11  driving_days             14999 non-null  int64  
     12  device                   14999 non-null  object 
    dtypes: float64(3), int64(8), object(2)
    memory usage: 1.5+ MB


**Question:** Are there any missing values in your data?

Yes, the label column is missing 700 values.

Use `.head()`.




```python
df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>label</th>
      <th>sessions</th>
      <th>drives</th>
      <th>total_sessions</th>
      <th>n_days_after_onboarding</th>
      <th>total_navigations_fav1</th>
      <th>total_navigations_fav2</th>
      <th>driven_km_drives</th>
      <th>duration_minutes_drives</th>
      <th>activity_days</th>
      <th>driving_days</th>
      <th>device</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>retained</td>
      <td>283</td>
      <td>226</td>
      <td>296.748273</td>
      <td>2276</td>
      <td>208</td>
      <td>0</td>
      <td>2628.845068</td>
      <td>1985.775061</td>
      <td>28</td>
      <td>19</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>retained</td>
      <td>133</td>
      <td>107</td>
      <td>326.896596</td>
      <td>1225</td>
      <td>19</td>
      <td>64</td>
      <td>13715.920550</td>
      <td>3160.472914</td>
      <td>13</td>
      <td>11</td>
      <td>iPhone</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>retained</td>
      <td>114</td>
      <td>95</td>
      <td>135.522926</td>
      <td>2651</td>
      <td>0</td>
      <td>0</td>
      <td>3059.148818</td>
      <td>1610.735904</td>
      <td>14</td>
      <td>8</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>retained</td>
      <td>49</td>
      <td>40</td>
      <td>67.589221</td>
      <td>15</td>
      <td>322</td>
      <td>7</td>
      <td>913.591123</td>
      <td>587.196542</td>
      <td>7</td>
      <td>3</td>
      <td>iPhone</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>retained</td>
      <td>84</td>
      <td>68</td>
      <td>168.247020</td>
      <td>1562</td>
      <td>166</td>
      <td>5</td>
      <td>3950.202008</td>
      <td>1219.555924</td>
      <td>27</td>
      <td>18</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>retained</td>
      <td>113</td>
      <td>103</td>
      <td>279.544437</td>
      <td>2637</td>
      <td>0</td>
      <td>0</td>
      <td>901.238699</td>
      <td>439.101397</td>
      <td>15</td>
      <td>11</td>
      <td>iPhone</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>retained</td>
      <td>3</td>
      <td>2</td>
      <td>236.725314</td>
      <td>360</td>
      <td>185</td>
      <td>18</td>
      <td>5249.172828</td>
      <td>726.577205</td>
      <td>28</td>
      <td>23</td>
      <td>iPhone</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>retained</td>
      <td>39</td>
      <td>35</td>
      <td>176.072845</td>
      <td>2999</td>
      <td>0</td>
      <td>0</td>
      <td>7892.052468</td>
      <td>2466.981741</td>
      <td>22</td>
      <td>20</td>
      <td>iPhone</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>retained</td>
      <td>57</td>
      <td>46</td>
      <td>183.532018</td>
      <td>424</td>
      <td>0</td>
      <td>26</td>
      <td>2651.709764</td>
      <td>1594.342984</td>
      <td>25</td>
      <td>20</td>
      <td>Android</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>churned</td>
      <td>84</td>
      <td>68</td>
      <td>244.802115</td>
      <td>2997</td>
      <td>72</td>
      <td>0</td>
      <td>6043.460295</td>
      <td>2341.838528</td>
      <td>7</td>
      <td>3</td>
      <td>iPhone</td>
    </tr>
  </tbody>
</table>
</div>



Use `.drop()` to remove the ID column since we don't need this information for your analysis.


```python
df = df.drop('ID', axis=1)
```

Now, check the class balance of the dependent (target) variable, `label`.


```python
df['label'].value_counts(normalize=True)
```




    retained    0.822645
    churned     0.177355
    Name: label, dtype: float64



Call `.describe()` on the data.



```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sessions</th>
      <th>drives</th>
      <th>total_sessions</th>
      <th>n_days_after_onboarding</th>
      <th>total_navigations_fav1</th>
      <th>total_navigations_fav2</th>
      <th>driven_km_drives</th>
      <th>duration_minutes_drives</th>
      <th>activity_days</th>
      <th>driving_days</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>80.633776</td>
      <td>67.281152</td>
      <td>189.964447</td>
      <td>1749.837789</td>
      <td>121.605974</td>
      <td>29.672512</td>
      <td>4039.340921</td>
      <td>1860.976012</td>
      <td>15.537102</td>
      <td>12.179879</td>
    </tr>
    <tr>
      <th>std</th>
      <td>80.699065</td>
      <td>65.913872</td>
      <td>136.405128</td>
      <td>1008.513876</td>
      <td>148.121544</td>
      <td>45.394651</td>
      <td>2502.149334</td>
      <td>1446.702288</td>
      <td>9.004655</td>
      <td>7.824036</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.220211</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>60.441250</td>
      <td>18.282082</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>23.000000</td>
      <td>20.000000</td>
      <td>90.661156</td>
      <td>878.000000</td>
      <td>9.000000</td>
      <td>0.000000</td>
      <td>2212.600607</td>
      <td>835.996260</td>
      <td>8.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>56.000000</td>
      <td>48.000000</td>
      <td>159.568115</td>
      <td>1741.000000</td>
      <td>71.000000</td>
      <td>9.000000</td>
      <td>3493.858085</td>
      <td>1478.249859</td>
      <td>16.000000</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>112.000000</td>
      <td>93.000000</td>
      <td>254.192341</td>
      <td>2623.500000</td>
      <td>178.000000</td>
      <td>43.000000</td>
      <td>5289.861262</td>
      <td>2464.362632</td>
      <td>23.000000</td>
      <td>19.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>743.000000</td>
      <td>596.000000</td>
      <td>1216.154633</td>
      <td>3500.000000</td>
      <td>1236.000000</td>
      <td>415.000000</td>
      <td>21183.401890</td>
      <td>15851.727160</td>
      <td>31.000000</td>
      <td>30.000000</td>
    </tr>
  </tbody>
</table>
</div>



**Question:** Are there any variables that could potentially have outliers just by assessing at the quartile values, standard deviation, and max values?

Yes, the following columns all seem to have outliers:

sessions,
drives,
total_sessions,
total_navigations_fav1,
total_navigations_fav2,
driven_km_drives,
duration_minutes_drives,

All of these columns have max values that are multiple standard deviations above the 75th percentile. This could indicate outliers in these variables.

### **Task 2b. Create features**

Create features that may be of interest to the stakeholder and/or that are needed to address the business scenario/problem.

#### **`km_per_driving_day`**

You know from earlier EDA that churn rate correlates with distance driven per driving day in the last month. It might be helpful to engineer a feature that captures this information.

1. Create a new column in `df` called `km_per_driving_day`, which represents the mean distance driven per driving day for each user.

2. Call the `describe()` method on the new column.


```python
# 1. Create `km_per_driving_day` column
df['km_per_driving_day'] = df['driven_km_drives'] / df['driving_days']

# 2. Call `describe()` on the new column
df['km_per_driving_day'].describe()
```




    count    1.499900e+04
    mean              inf
    std               NaN
    min      3.022063e+00
    25%      1.672804e+02
    50%      3.231459e+02
    75%      7.579257e+02
    max               inf
    Name: km_per_driving_day, dtype: float64



Note that some values are infinite. This is the result of there being values of zero in the `driving_days` column. Pandas imputes a value of infinity in the corresponding rows of the new column because division by zero is undefined.

1. Convert these values from infinity to zero. You can use `np.inf` to refer to a value of infinity.

2. Call `describe()` on the `km_per_driving_day` column to verify that it worked.


```python
# 1. Convert infinite values to zero
df.loc[df['km_per_driving_day']==np.inf, 'km_per_driving_day'] = 0

# 2. Confirm that it worked
df['km_per_driving_day'].describe()
```




    count    14999.000000
    mean       578.963113
    std       1030.094384
    min          0.000000
    25%        136.238895
    50%        272.889272
    75%        558.686918
    max      15420.234110
    Name: km_per_driving_day, dtype: float64



#### **`professional_driver`**

Create a new, binary feature called `professional_driver` that is a 1 for users who had 60 or more drives <u>**and**</u> drove on 15+ days in the last month.

**Note:** The objective is to create a new feature that separates professional drivers from other drivers. In this scenario, domain knowledge and intuition are used to determine these deciding thresholds, but ultimately they are arbitrary.

To create this column, use the [`np.where()`](https://numpy.org/doc/stable/reference/generated/numpy.where.html) function. This function accepts as arguments:
1. A condition
2. What to return when the condition is true
3. What to return when the condition is false

```
Example:
x = [1, 2, 3]
x = np.where(x > 2, 100, 0)
x
array([  0,   0, 100])
```


```python
# Create `professional_driver` column
df['professional_driver'] = np.where((df['drives'] >= 60) & (df['driving_days'] >= 15), 1, 0)
```

Perform a quick inspection of the new variable.

1. Check the count of professional drivers and non-professionals

2. Within each class (professional and non-professional) calculate the churn rate


```python
# 1. Check count of professionals and non-professionals
print(df['professional_driver'].value_counts())

# 2. Check in-class churn rate
df.groupby(['professional_driver'])['label'].value_counts(normalize=True)
```

    0    12405
    1     2594
    Name: professional_driver, dtype: int64





    professional_driver  label   
    0                    retained    0.801202
                         churned     0.198798
    1                    retained    0.924437
                         churned     0.075563
    Name: label, dtype: float64



The churn rate for professional drivers is 7.6%, while the churn rate for non-professionals is 19.9%. This seems like it could add predictive signal to the model.

<img src="images/Construct.png" width="100" height="100" align=left>

## **PACE: Construct**

After analysis and deriving variables with close relationships, it is time to begin constructing the model.

Consider the questions in your PACE Strategy Document to reflect on the Construct stage.

In this stage, consider the following question:

* Why did you select the X variables you did?

At first, I dropped columns that showed high multicollinearity. Later on, I fine-tuned my variable selection by running and rerunning models to see how the changes affected accuracy, recall, and precision. My initial choice of variables was guided by the business objective and the insights I gained from my earlier EDA.

### **Task 3a. Preparing variables**

Call `info()` on the dataframe to check the data type of the `label` variable and to verify if there are any missing values.


```python
df.info()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 14999 entries, 0 to 14998
    Data columns (total 14 columns):
     #   Column                   Non-Null Count  Dtype  
    ---  ------                   --------------  -----  
     0   label                    14299 non-null  object 
     1   sessions                 14999 non-null  int64  
     2   drives                   14999 non-null  int64  
     3   total_sessions           14999 non-null  float64
     4   n_days_after_onboarding  14999 non-null  int64  
     5   total_navigations_fav1   14999 non-null  int64  
     6   total_navigations_fav2   14999 non-null  int64  
     7   driven_km_drives         14999 non-null  float64
     8   duration_minutes_drives  14999 non-null  float64
     9   activity_days            14999 non-null  int64  
     10  driving_days             14999 non-null  int64  
     11  device                   14999 non-null  object 
     12  km_per_driving_day       14999 non-null  float64
     13  professional_driver      14999 non-null  int64  
    dtypes: float64(4), int64(8), object(2)
    memory usage: 1.6+ MB


Because you know from previous EDA that there is no evidence of a non-random cause of the 700 missing values in the `label` column, and because these observations comprise less than 5% of the data, use the `dropna()` method to drop the rows that are missing this data.


```python
# Drop rows with missing data in `label` column
df = df.dropna(subset=['label'])
```

#### **Impute outliers**

You rarely want to drop outliers, and generally will not do so unless there is a clear reason for it (e.g., typographic errors).

At times outliers can be changed to the **median, mean, 95th percentile, etc.**

Previously, you determined that seven of the variables had clear signs of containing outliers:

* `sessions`
* `drives`
* `total_sessions`
* `total_navigations_fav1`
* `total_navigations_fav2`
* `driven_km_drives`
* `duration_minutes_drives`

For this analysis, impute the outlying values for these columns. Calculate the **95th percentile** of each column and change to this value any value in the column that exceeds it.



```python
# Impute outliers
for column in ['sessions', 'drives', 'total_sessions', 'total_navigations_fav1',
               'total_navigations_fav2', 'driven_km_drives', 'duration_minutes_drives']:
    threshold = df[column].quantile(0.95)
    df.loc[df[column] > threshold, column] = threshold
```

Call `describe()`.


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sessions</th>
      <th>drives</th>
      <th>total_sessions</th>
      <th>n_days_after_onboarding</th>
      <th>total_navigations_fav1</th>
      <th>total_navigations_fav2</th>
      <th>driven_km_drives</th>
      <th>duration_minutes_drives</th>
      <th>activity_days</th>
      <th>driving_days</th>
      <th>km_per_driving_day</th>
      <th>professional_driver</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>14299.000000</td>
      <td>14299.000000</td>
      <td>14299.000000</td>
      <td>14299.000000</td>
      <td>14299.000000</td>
      <td>14299.000000</td>
      <td>14299.000000</td>
      <td>14299.000000</td>
      <td>14299.000000</td>
      <td>14299.000000</td>
      <td>14299.000000</td>
      <td>14299.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>76.539688</td>
      <td>63.964683</td>
      <td>183.717304</td>
      <td>1751.822505</td>
      <td>114.562767</td>
      <td>27.187216</td>
      <td>3944.558631</td>
      <td>1792.911210</td>
      <td>15.544653</td>
      <td>12.182530</td>
      <td>581.942399</td>
      <td>0.173998</td>
    </tr>
    <tr>
      <th>std</th>
      <td>67.243178</td>
      <td>55.127927</td>
      <td>118.720520</td>
      <td>1008.663834</td>
      <td>124.378550</td>
      <td>36.715302</td>
      <td>2218.358258</td>
      <td>1224.329759</td>
      <td>9.016088</td>
      <td>7.833835</td>
      <td>1038.254509</td>
      <td>0.379121</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.220211</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>60.441250</td>
      <td>18.282082</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>23.000000</td>
      <td>20.000000</td>
      <td>90.457733</td>
      <td>878.500000</td>
      <td>10.000000</td>
      <td>0.000000</td>
      <td>2217.319909</td>
      <td>840.181344</td>
      <td>8.000000</td>
      <td>5.000000</td>
      <td>136.168003</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>56.000000</td>
      <td>48.000000</td>
      <td>158.718571</td>
      <td>1749.000000</td>
      <td>71.000000</td>
      <td>9.000000</td>
      <td>3496.545617</td>
      <td>1479.394387</td>
      <td>16.000000</td>
      <td>12.000000</td>
      <td>273.301012</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>111.000000</td>
      <td>93.000000</td>
      <td>253.540450</td>
      <td>2627.500000</td>
      <td>178.000000</td>
      <td>43.000000</td>
      <td>5299.972162</td>
      <td>2466.928876</td>
      <td>23.000000</td>
      <td>19.000000</td>
      <td>558.018761</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>243.000000</td>
      <td>200.000000</td>
      <td>455.439492</td>
      <td>3500.000000</td>
      <td>422.000000</td>
      <td>124.000000</td>
      <td>8898.716275</td>
      <td>4668.180092</td>
      <td>31.000000</td>
      <td>30.000000</td>
      <td>15420.234110</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### **Encode categorical variables**

Change the data type of the `label` column to be binary. This change is needed to train a logistic regression model.

Assign a `0` for all `retained` users.

Assign a `1` for all `churned` users.

Save this variable as `label2` as to not overwrite the original `label` variable.

**Note:** There are many ways to do this. Consider using `np.where()` as you did earlier in this notebook.


```python
# Create binary `label2` column
df['label2'] = np.where(df['label']=='churned', 1, 0)
df[['label', 'label2']].tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>label2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14994</th>
      <td>retained</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14995</th>
      <td>retained</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14996</th>
      <td>retained</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14997</th>
      <td>churned</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14998</th>
      <td>retained</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### **Task 3b. Determine whether assumptions have been met**

The following are the assumptions for logistic regression:

* Independent observations (This refers to how the data was collected.)

* No extreme outliers

* Little to no multicollinearity among X predictors

* Linear relationship between X and the **logit** of y

For the first assumption, you can assume that observations are independent for this project.

The second assumption has already been addressed.

The last assumption will be verified after modeling.

**Note:** In practice, modeling assumptions are often violated, and depending on the specifics of your use case and the severity of the violation, it might not affect your model much at all or it will result in a failed model.

#### **Collinearity**

Check the correlation among predictor variables. First, generate a correlation matrix.


```python
# Generate a correlation matrix
df.corr(method='pearson')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sessions</th>
      <th>drives</th>
      <th>total_sessions</th>
      <th>n_days_after_onboarding</th>
      <th>total_navigations_fav1</th>
      <th>total_navigations_fav2</th>
      <th>driven_km_drives</th>
      <th>duration_minutes_drives</th>
      <th>activity_days</th>
      <th>driving_days</th>
      <th>km_per_driving_day</th>
      <th>professional_driver</th>
      <th>label2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sessions</th>
      <td>1.000000</td>
      <td>0.996942</td>
      <td>0.597189</td>
      <td>0.007101</td>
      <td>0.001858</td>
      <td>0.008536</td>
      <td>0.002996</td>
      <td>-0.004545</td>
      <td>0.025113</td>
      <td>0.020294</td>
      <td>-0.011569</td>
      <td>0.443654</td>
      <td>0.034911</td>
    </tr>
    <tr>
      <th>drives</th>
      <td>0.996942</td>
      <td>1.000000</td>
      <td>0.595285</td>
      <td>0.006940</td>
      <td>0.001058</td>
      <td>0.009505</td>
      <td>0.003445</td>
      <td>-0.003889</td>
      <td>0.024357</td>
      <td>0.019608</td>
      <td>-0.010989</td>
      <td>0.444425</td>
      <td>0.035865</td>
    </tr>
    <tr>
      <th>total_sessions</th>
      <td>0.597189</td>
      <td>0.595285</td>
      <td>1.000000</td>
      <td>0.006596</td>
      <td>0.000187</td>
      <td>0.010371</td>
      <td>0.001016</td>
      <td>-0.000338</td>
      <td>0.015755</td>
      <td>0.012953</td>
      <td>-0.016167</td>
      <td>0.254433</td>
      <td>0.024568</td>
    </tr>
    <tr>
      <th>n_days_after_onboarding</th>
      <td>0.007101</td>
      <td>0.006940</td>
      <td>0.006596</td>
      <td>1.000000</td>
      <td>-0.002450</td>
      <td>-0.004968</td>
      <td>-0.004652</td>
      <td>-0.010167</td>
      <td>-0.009418</td>
      <td>-0.007321</td>
      <td>0.011764</td>
      <td>0.003770</td>
      <td>-0.129263</td>
    </tr>
    <tr>
      <th>total_navigations_fav1</th>
      <td>0.001858</td>
      <td>0.001058</td>
      <td>0.000187</td>
      <td>-0.002450</td>
      <td>1.000000</td>
      <td>0.002866</td>
      <td>-0.007368</td>
      <td>0.005646</td>
      <td>0.010902</td>
      <td>0.010419</td>
      <td>-0.000197</td>
      <td>-0.000224</td>
      <td>0.052322</td>
    </tr>
    <tr>
      <th>total_navigations_fav2</th>
      <td>0.008536</td>
      <td>0.009505</td>
      <td>0.010371</td>
      <td>-0.004968</td>
      <td>0.002866</td>
      <td>1.000000</td>
      <td>0.003559</td>
      <td>-0.003009</td>
      <td>-0.004425</td>
      <td>0.002000</td>
      <td>0.006751</td>
      <td>0.007126</td>
      <td>0.015032</td>
    </tr>
    <tr>
      <th>driven_km_drives</th>
      <td>0.002996</td>
      <td>0.003445</td>
      <td>0.001016</td>
      <td>-0.004652</td>
      <td>-0.007368</td>
      <td>0.003559</td>
      <td>1.000000</td>
      <td>0.690515</td>
      <td>-0.007441</td>
      <td>-0.009549</td>
      <td>0.344811</td>
      <td>-0.000904</td>
      <td>0.019767</td>
    </tr>
    <tr>
      <th>duration_minutes_drives</th>
      <td>-0.004545</td>
      <td>-0.003889</td>
      <td>-0.000338</td>
      <td>-0.010167</td>
      <td>0.005646</td>
      <td>-0.003009</td>
      <td>0.690515</td>
      <td>1.000000</td>
      <td>-0.007895</td>
      <td>-0.009425</td>
      <td>0.239627</td>
      <td>-0.012128</td>
      <td>0.040407</td>
    </tr>
    <tr>
      <th>activity_days</th>
      <td>0.025113</td>
      <td>0.024357</td>
      <td>0.015755</td>
      <td>-0.009418</td>
      <td>0.010902</td>
      <td>-0.004425</td>
      <td>-0.007441</td>
      <td>-0.007895</td>
      <td>1.000000</td>
      <td>0.947687</td>
      <td>-0.397433</td>
      <td>0.453825</td>
      <td>-0.303851</td>
    </tr>
    <tr>
      <th>driving_days</th>
      <td>0.020294</td>
      <td>0.019608</td>
      <td>0.012953</td>
      <td>-0.007321</td>
      <td>0.010419</td>
      <td>0.002000</td>
      <td>-0.009549</td>
      <td>-0.009425</td>
      <td>0.947687</td>
      <td>1.000000</td>
      <td>-0.407917</td>
      <td>0.469776</td>
      <td>-0.294259</td>
    </tr>
    <tr>
      <th>km_per_driving_day</th>
      <td>-0.011569</td>
      <td>-0.010989</td>
      <td>-0.016167</td>
      <td>0.011764</td>
      <td>-0.000197</td>
      <td>0.006751</td>
      <td>0.344811</td>
      <td>0.239627</td>
      <td>-0.397433</td>
      <td>-0.407917</td>
      <td>1.000000</td>
      <td>-0.165966</td>
      <td>0.148583</td>
    </tr>
    <tr>
      <th>professional_driver</th>
      <td>0.443654</td>
      <td>0.444425</td>
      <td>0.254433</td>
      <td>0.003770</td>
      <td>-0.000224</td>
      <td>0.007126</td>
      <td>-0.000904</td>
      <td>-0.012128</td>
      <td>0.453825</td>
      <td>0.469776</td>
      <td>-0.165966</td>
      <td>1.000000</td>
      <td>-0.122312</td>
    </tr>
    <tr>
      <th>label2</th>
      <td>0.034911</td>
      <td>0.035865</td>
      <td>0.024568</td>
      <td>-0.129263</td>
      <td>0.052322</td>
      <td>0.015032</td>
      <td>0.019767</td>
      <td>0.040407</td>
      <td>-0.303851</td>
      <td>-0.294259</td>
      <td>0.148583</td>
      <td>-0.122312</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Now, plot a correlation heatmap.


```python
# Plot correlation heatmap
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(method='pearson'), vmin=-1, vmax=1, annot=True, cmap='coolwarm')
plt.title('Correlation heatmap indicates many low correlated variables',
          fontsize=18)
plt.show();
```


![png](output_57_0.png)


If there are predictor variables that have a Pearson correlation coefficient value greater than the **absolute value of 0.7**, these variables are strongly multicollinear. Therefore, only one of these variables should be used in your model.

**Note:** 0.7 is an arbitrary threshold. Some industries may use 0.6, 0.8, etc.

**Question:** Which variables are multicollinear with each other?

sessions and drives: 1.0

driving_days and activity_days: 0.95

### **Task 3c. Create dummies (if necessary)**

If you have selected `device` as an X variable, you will need to create dummy variables since this variable is categorical.

In cases with many categorical variables, you can use pandas built-in [`pd.get_dummies()`](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html), or you can use scikit-learn's [`OneHotEncoder()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) function.

**Note:** Variables with many categories should only be dummied if absolutely necessary. Each category will result in a coefficient for your model which can lead to overfitting.

Because this dataset only has one remaining categorical feature (`device`), it's not necessary to use one of these special functions. You can just implement the transformation directly.

Create a new, binary column called `device2` that encodes user devices as follows:

* `Android` -> `0`
* `iPhone` -> `1`


```python
# Create new `device2` variable
df['device2'] = np.where(df['device']=='Android', 0, 1)
df[['device', 'device2']].tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>device</th>
      <th>device2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14994</th>
      <td>iPhone</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14995</th>
      <td>Android</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14996</th>
      <td>iPhone</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14997</th>
      <td>iPhone</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14998</th>
      <td>iPhone</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### **Task 3d. Model building**

#### **Assign predictor variables and target**

To build your model you need to determine what X variables you want to include in your model to predict your target&mdash;`label2`.

Drop the following variables and assign the results to `X`:

* `label` (this is the target)
* `label2` (this is the target)
* `device` (this is the non-binary-encoded categorical variable)
* `sessions` (this had high multicollinearity)
* `driving_days` (this had high multicollinearity)

**Note:** Notice that `sessions` and `driving_days` were selected to be dropped, rather than `drives` and `activity_days`. The reason for this is that the features that were kept for modeling had slightly stronger correlations with the target variable than the features that were dropped.


```python
# Isolate predictor variables
X = df.drop(columns = ['label', 'label2', 'device', 'sessions', 'driving_days'])
```

Now, isolate the dependent (target) variable. Assign it to a variable called `y`.


```python
# Isolate target variable
y = df['label2']
```

#### **Split the data**

Use scikit-learn's [`train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function to perform a train/test split on your data using the X and y variables you assigned above.

**Note 1:** It is important to do a train test to obtain accurate predictions.  You always want to fit your model on your training set and evaluate your model on your test set to avoid data leakage.

**Note 2:** Because the target class is imbalanced (82% retained vs. 18% churned), you want to make sure that you don't get an unlucky split that over- or under-represents the frequency of the minority class. Set the function's `stratify` parameter to `y` to ensure that the minority class appears in both train and test sets in the same proportion that it does in the overall dataset.


```python
# Perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
```


```python
# Use .head()
X_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>drives</th>
      <th>total_sessions</th>
      <th>n_days_after_onboarding</th>
      <th>total_navigations_fav1</th>
      <th>total_navigations_fav2</th>
      <th>driven_km_drives</th>
      <th>duration_minutes_drives</th>
      <th>activity_days</th>
      <th>km_per_driving_day</th>
      <th>professional_driver</th>
      <th>device2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>152</th>
      <td>108</td>
      <td>186.192746</td>
      <td>3116</td>
      <td>243</td>
      <td>124</td>
      <td>8898.716275</td>
      <td>4668.180092</td>
      <td>24</td>
      <td>612.305861</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11899</th>
      <td>2</td>
      <td>3.487590</td>
      <td>794</td>
      <td>114</td>
      <td>18</td>
      <td>3286.545691</td>
      <td>1780.902733</td>
      <td>5</td>
      <td>3286.545691</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10937</th>
      <td>139</td>
      <td>347.106403</td>
      <td>331</td>
      <td>4</td>
      <td>7</td>
      <td>7400.838975</td>
      <td>2349.305267</td>
      <td>15</td>
      <td>616.736581</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>669</th>
      <td>108</td>
      <td>455.439492</td>
      <td>2320</td>
      <td>11</td>
      <td>4</td>
      <td>6566.424830</td>
      <td>4558.459870</td>
      <td>18</td>
      <td>410.401552</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8406</th>
      <td>10</td>
      <td>89.475821</td>
      <td>2478</td>
      <td>135</td>
      <td>0</td>
      <td>1271.248661</td>
      <td>938.711572</td>
      <td>27</td>
      <td>74.779333</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Use scikit-learn to instantiate a logistic regression model. Add the argument `penalty = None`.

It is important to add `penalty = None` since your predictors are unscaled.

Refer to scikit-learn's [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) documentation for more information.

Fit the model on `X_train` and `y_train`.


```python
model = LogisticRegression(penalty='none', max_iter=400)

model.fit(X_train, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=400,
                       multi_class='auto', n_jobs=None, penalty='none',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)



Call the `.coef_` attribute on the model to get the coefficients of each variable.  The coefficients are in order of how the variables are listed in the dataset.  Remember that the coefficients represent the change in the **log odds** of the target variable for **every one unit increase in X**.

If you want, create a series whose index is the column names and whose values are the coefficients in `model.coef_`.


```python
pd.Series(model.coef_[0], index=X.columns)
```




    drives                     0.001913
    total_sessions             0.000327
    n_days_after_onboarding   -0.000406
    total_navigations_fav1     0.001232
    total_navigations_fav2     0.000931
    driven_km_drives          -0.000015
    duration_minutes_drives    0.000109
    activity_days             -0.106032
    km_per_driving_day         0.000018
    professional_driver       -0.001529
    device2                   -0.001041
    dtype: float64



Call the model's `intercept_` attribute to get the intercept of the model.


```python
model.intercept_
```




    array([-0.00170675])



#### **Check final assumption**

Verify the linear relationship between X and the estimated log odds (known as logits) by making a regplot.

Call the model's `predict_proba()` method to generate the probability of response for each sample in the training data. (The training data is the argument to the method.) Assign the result to a variable called `training_probabilities`. This results in a 2-D array where each row represents a user in `X_train`. The first column is the probability of the user not churning, and the second column is the probability of the user churning.


```python
# Get the predicted probabilities of the training data
training_probabilities = model.predict_proba(X_train)
training_probabilities
```




    array([[0.93963483, 0.06036517],
           [0.61967304, 0.38032696],
           [0.76463181, 0.23536819],
           ...,
           [0.91909641, 0.08090359],
           [0.85092112, 0.14907888],
           [0.93516293, 0.06483707]])



In logistic regression, the relationship between a predictor variable and the dependent variable does not need to be linear, however, the log-odds (a.k.a., logit) of the dependent variable with respect to the predictor variable should be linear. Here is the formula for calculating log-odds, where _p_ is the probability of response:
<br>
$$
logit(p) = ln(\frac{p}{1-p})
$$
<br>

1. Create a dataframe called `logit_data` that is a copy of `df`.

2. Create a new column called `logit` in the `logit_data` dataframe. The data in this column should represent the logit for each user.



```python
# 1. Copy the `X_train` dataframe and assign to `logit_data`
logit_data = X_train.copy()

# 2. Create a new `logit` column in the `logit_data` df
logit_data['logit'] = [np.log(prob[1] / prob[0]) for prob in training_probabilities]
```

Plot a regplot where the x-axis represents an independent variable and the y-axis represents the log-odds of the predicted probabilities.

In an exhaustive analysis, this would be plotted for each continuous or discrete predictor variable. Here we show only `driving_days`.


```python
# Plot regplot of `activity_days` log-odds
sns.regplot(x='activity_days', y='logit', data=logit_data, scatter_kws={'s': 2, 'alpha': 0.5})
plt.title('Log-odds: activity_days');
```


![png](output_81_0.png)


<img src="images/Execute.png" width="100" height="100" align=left>

## **PACE: Execute**

Consider the questions in your PACE Strategy Document to reflect on the Execute stage.

### **Task 4a. Results and evaluation**

If the logistic assumptions are met, the model results can be appropriately interpreted.

Use the code block below to make predictions on the test data.



```python
# Generate predictions on X_test
y_preds = model.predict(X_test)
```

Now, use the `score()` method on the model with `X_test` and `y_test` as its two arguments. The default score in scikit-learn is **accuracy**.  What is the accuracy of your model?

*Consider:  Is accuracy the best metric to use to evaluate this model?*


```python
# Score the model (accuracy) on the test data
model.score(X_test, y_test)
```




    0.8237762237762237



### **Task 4b. Show results with a confusion matrix**

Use the `confusion_matrix` function to obtain a confusion matrix. Use `y_test` and `y_preds` as arguments.


```python
cm = confusion_matrix(y_test, y_preds)
```

Next, use the `ConfusionMatrixDisplay()` function to display the confusion matrix from the above cell, passing the confusion matrix you just created as its argument.


```python
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                              display_labels=['retained', 'churned'],
                              )
disp.plot();
```


![png](output_91_0.png)


You can use the confusion matrix to compute precision and recall manually. You can also use scikit-learn's [`classification_report()`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) function to generate a table from `y_test` and `y_preds`.


```python
# Calculate precision manually
precision = cm[1,1] / (cm[0, 1] + cm[1, 1])
precision
```




    0.5178571428571429




```python
# Calculate recall manually
recall = cm[1,1] / (cm[1, 0] + cm[1, 1])
recall
```




    0.0914826498422713




```python
# Create a classification report
target_labels = ['retained', 'churned']
print(classification_report(y_test, y_preds, target_names=target_labels))
```

                  precision    recall  f1-score   support
    
        retained       0.83      0.98      0.90      2941
         churned       0.52      0.09      0.16       634
    
        accuracy                           0.82      3575
       macro avg       0.68      0.54      0.53      3575
    weighted avg       0.78      0.82      0.77      3575
    


**Note:** The model has decent precision but very low recall, which means that it makes a lot of false negative predictions and fails to capture users who will churn.

### **BONUS**

Generate a bar graph of the model's coefficients for a visual representation of the importance of the model's features.


```python
# Create a list of (column_name, coefficient) tuples
feature_importance = list(zip(X_train.columns, model.coef_[0]))

# Sort the list by coefficient value
feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
feature_importance
```




    [('drives', 0.001913369447769776),
     ('total_navigations_fav1', 0.001231754741616306),
     ('total_navigations_fav2', 0.0009314786513814626),
     ('total_sessions', 0.00032707088819142904),
     ('duration_minutes_drives', 0.00010909343558951453),
     ('km_per_driving_day', 1.8223094015325207e-05),
     ('driven_km_drives', -1.4860453424647997e-05),
     ('n_days_after_onboarding', -0.00040647763730561445),
     ('device2', -0.0010412175209008018),
     ('professional_driver', -0.0015285041567402024),
     ('activity_days', -0.10603196504385491)]




```python
# Plot the feature importances
import seaborn as sns
sns.barplot(x=[x[1] for x in feature_importance],
            y=[x[0] for x in feature_importance],
            orient='h')
plt.title('Feature importance');
```


![png](output_99_0.png)


### **Task 4c. Conclusion**

Now that you've built your regression model, the next step is to share your findings with the Waze leadership team. Consider the following questions as you prepare to write your executive summary. Think about key points you may want to share with the team, and what information is most relevant to the user churn project.

**Questions:**

1. What variable most influenced the model's prediction? How? Was this surprising?

2. Were there any variables that you expected to be stronger predictors than they were?

3. Why might a variable you thought to be important not be important in the model?

4. Would you recommend that Waze use this model? Why or why not?

5. What could you do to improve this model?

6. What additional features would you like to have to help improve the model?


What variable most influenced the model's prediction? How? Was this surprising?

The variable that most influenced my model’s predictions was activity_days. It had a negative correlation with user churn. This wasn’t surprising to me, since activity_days was strongly correlated with driving_days, which I already knew from my EDA to have a negative correlation with churn.

Were there any variables that you expected to be stronger predictors than they were?

Yes. Based on my earlier EDA, I expected km_per_driving_day to be a much stronger predictor. I saw that user churn rate increased as km_per_driving_day increased, and in the correlation heatmap it actually showed the strongest positive correlation with churn out of all predictor variables. But in the model, it ended up being the second-least-important variable.

Why might a variable I thought would be important not be important in the model?

In multiple logistic regression, features can interact with each other in ways that create counterintuitive outcomes. That’s what likely happened here. This is both a strength and a limitation of predictive models: capturing these interactions often improves predictive power, but it can also make the model harder to interpret.

Would I recommend that Waze use this model? Why or why not?

It depends on the purpose. If Waze were to use this model for important business decisions, I would not recommend it, because its recall score shows it’s not a strong enough predictor. However, if the goal is to guide exploratory analysis and generate insights for further testing, then yes—it could still be valuable.

What could I do to improve this model?

I could engineer new features to strengthen the predictive signal, especially with the benefit of domain knowledge. For instance, one of my engineered features, professional_driver, turned out to be the third-most-important predictor. It may also help to scale the predictor variables and/or reconstruct the model with different combinations of predictors to reduce noise from weaker features.

What additional features would I like to have to help improve the model?

I would really like to have drive-level information for each user, such as drive times or geographic locations. More granular app usage data would also be useful. For example, how often users report or confirm road hazard alerts. Another valuable addition would be the monthly count of unique starting and ending locations per driver.

**Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged. 
