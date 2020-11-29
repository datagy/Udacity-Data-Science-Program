# Starbucks Capstone Project
## Introduction
For the capstone project, I selected the Starbucks Challenge. The dataset provided contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks.

Not all users receive the same offer, and that is the challenge to solve with this data set.

## Goal of Analysis
The end goal of the analysis was to determine which features have the most impact on determining whether a user completes and offer or not. We'll take this analysis one step further and see if there are variations between different offer types.

At the end of our overview, we'll have a strong overview of which features are most important in determining a successul offer completion.

We'll use a random forest classifier to determine which features matter most.

![random forest](https://github.com/datagy/Udacity-Data-Science-Program/blob/master/Capstone/images/tree.png)

## Data Sets Overview
The data is contained in three files:

* portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
* profile.json - demographic data for each customer
* transcript.json - records for transactions, offers received, offers viewed, and offers completed

### Data Cleaning
The data were of differing degrees of data cleanliness. For example, the `portfolio` dataset required very little cleaning. Others required more intensive cleaning, where data was structured as nested dictionaries within the resulting dataframes. Functions were developed to properly clean all data, one-hot encoding data as necessary.

### Dealing with Missing Values
One important consideration with any machine learning project is determining what to do with values. There are a number of typically considered options:

1. Drop missing values row-wise,
2. Drop missing values column-wise,
3. Impute missing values.

In our case, we encountered missing values in our `Profile` dataset. The values existed only in the `age` and `income` columns. We don't know enough to impute the values (and this analysis was outside the scope of the project). Because of this, it's best to drop those records. We'll drop the records, rather than the columns. If we had determined that these features had significant influence on our models, then we could return and impute the values.

### Determining Offer Success
In order to determine whether or not an offer was successul, some manipulation needed to be done to the cleaned `transcript` dataset. In particular, the decision was made to classify and offer as successful if and only if the user viewed the offer and completed the offer.

The following logic was applied to the `cleaned_transcript` dataframe:

```python
transcript_success = pd.pivot_table(data=cleaned_transcript, index=['person', 'offer_id'], columns=['event'], aggfunc='count').reset_index()
transcript_success.columns = ['person', 'offer_id', 'offer completed', 'offer received', 'offer viewed']
transcript_success['offer completed'] = transcript_success['offer completed'].apply(lambda x: 1 if x >= 1 else 0)
transcript_success['offer viewed'] = transcript_success['offer viewed'].apply(lambda x: 1 if x >= 1 else 0)
transcript_success['success'] = transcript_success['offer completed'] + transcript_success['offer viewed']
transcript_success['success'] = transcript_success['success'].apply(lambda x: 1 if x == 2 else 0)
transcript_success = transcript_success.drop(columns = ['offer completed', 'offer viewed', 'offer received'])
```

This gave us a unique `success` indicator for each person-offer pair. A 0 indicated that the offer was not successful, while a 1 indicated that the offer was successful.

