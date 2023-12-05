# ML_Project

### AI &amp; Machine Learning group project at LUISS Fall 2023

###### Philip Fabrelius & Jan Stein

### Introduction
 
The project aims to help the leading e-commerce platform ShopEasy to provide personalized user experiences, special promotions and improved services to their customers by understanding the buying habits and behaviors of their customers. This is done by applying segmentation to an extensive dataset provided by the platform.

To successfully carry out the task at hand, the following steps were performed:  
- Exploratory Data Analysis (EDA)
- Preprocessing of data
- Testing of different clustering models
- Comparison between different clustering models
- Description of results

We identified two key variables for our segmentation:  
- **Monthly Paid** - average amount spent on ShopEasy per month
- **Average Item Cost** - average price of items purchased by the customer

Our best clustering model returned the following five segments:
- **Segment 1** - description
- **Segment 2** - description
- **Segment 3** - description
- **Segment 4** - description
- **Segment 5** - description

### Methods

#### Imported Libraries
- Pandas
- Seaborn
- Numpy
- Matplotlib
- Scikit learn - great library for preprocessing and machine learning algorithms.

    - StandardScaler
    - KMeans
    - silhouette_score
    - DBSCAN
    - AgglomerativeClustering
    - GaussianMixture

- Scipy - for calculation and visualization of hierarchical clustering.

    - Dendogram
    - Linkage

#### EDA

**Description of the Dataset**  
We assume all costs and other variables related to money presented as number of dollars.

- **personId**: Unique identifier for each user on the platform
- **accountTotal**: Total amount spent by the user on ShopEasy since their registration
- **frequencyIndex**: Reflects how frequently the user shops, with 1 being very frequent and values less than 1 being less frequent
- **itemCosts**: Total costs of items purchased by the user
- **singleItemCosts**: Costs of items that the user bought in a single purchase without opting for installments
- **multipleItemCosts**: Costs of items that the user decided to buy in installments
- **emergencyFunds**: Amount that the user decided to keep as a backup in their ShopEasy wallet for faster checkout or emergency purchases
- **itemBuyFrequency**: Frequency with which the user makes purchases
- **singleItemBuyFrequency**: How often the user makes single purchases without opting for installments
- **multipleItemBuyFrequency**: How often the user opts for installment-based purchases
- **emergencyUseFrequency**: How frequently the user taps into their emergency funds
- **emergencyCount**: Number of times the user has used their emergency funds
- **itemCount**: Total number of individual items purchased by the user
- **maxSpendLimit**: The maximum amount the user can spend in a single purchase, set by ShopEasy based on user's buying behavior and loyalty
- **monthlyPaid**: Total amount paid by the user every month
- **leastAmountPaid**: The least amount paid by the user in a single transaction
- **paymentCompletionRate**: Percentage of purchases where the user has paid the full amount
- **accountLifespan**: Duration for which the user has been registered on ShopEasy (in months)
- **location**: User's city or region
- **accountType**: The type of account held by the user. Regular for most users, Premium for those who have subscribed to ShopEasy premium services, and Student for users who have registered with a student ID
- **webUsage**: A metric (0-100) indicating the frequency with which the user shops on ShopEasy via web browsers. A higher number indicates more frequent web usage

**Scope**

For this assignment we're interested in investigating the relationship between the amount of money spent and the volume of products purchased, to better understand if customers are high/low value customers because of the cost of items purchased, or because of the amount of (cheap?) items purchased.

Therefore, and because of the size of the dataset, we have chosen to eliminate variables that are not relevant for this investigation.

**Columns to be cut**

1. **paymentCompletionRate:** because it is a payment option related variable
2. **maxSpendLimit:** because it is already represented through spending, which is represented in monthly paid
3. **emergencyCount:** because it just reflects the customers preferred payment option, which is irrelevant for this investigation
4. **emergencyUseFrequency:** because of the same reason
5. **emergencyFunds:** because of the same reason
6. **singleItemCosts:** because they are only related to payment option
7. **MultipleItemCosts:** because they are only related to payment option
8. **singleItemBuyFrequency:** because they are only related to payment option
9. **multipleItemBuyFrequency:** because they are only related to payment option
10. **personId:** This variable is just an internal customer ID and does not matter for our purposes

**Descriptive Statistics**

We used the following methods to learn more about the data:

-Describe() method - to get more information about range of values, mean, standard deviatio, number of observations and distribution.
- Info() method - to get a concise summary of the DataFrame. This method is useful for quickly understanding the structure of the dataset. Using info(), you can quickly assess which columns may require type conversion or additional preprocessing due to null values or incorrect data types.
- IsNull().sum() method - The isnull().sum() method is a two-step operation specifically geared towards identifying missing values in the DataFrame. This method is particularly useful for data cleaning and preprocessing, as handling missing values is a critical step in preparing data for analysis or modeling.

**Actions**

From our base set of 8 950 rows, we identified some columns (AccountTotal, ItemCost, ItemCount, MonthlyPaid) with minimum values of 0 where we believe that the value should be > 0 to be relevant data points. We therefore decided to drop these rows, 2 258 dropped rows.  
32 missing values where found for leastAmountPaid and they were also dropped. 6 660 rows remain, 74.4% of the original data.

**Feature Engineering**

The dataset includes variables for total amount of money spent (accountTotal), total amount of items purchased (itemCount), total cost of those items (itemCosts) and an average monthly spending variable (monthlyPaid). It does however not include a variable for the average item cost.

We add a variable for average item cost (**avgItemCost**) by dividing the itemCosts by itemCount.

**Analyzing Outliers**

We analyzed outliers in the dataset using a boxplot. Most columns have many distant outliers except the frequency variables created by ShopEasy, which make sense as they are a relative range variable between 0-1 or 0-100. Valuable information for later steps.

**Analyzing Categorical Variables**

Both categorical variables consists of three evenly distributed values. Location (New York, Los Angeles and Chicago) and AccountType (Premium, Student and Standard).

**Analyzing Numerical Features**

As a part of the univariate analysis with the main purpose of describing and finding patterns for one variable, we used histograms to analyze the distribution of all remaining numerical features, here are some findings:

- **accountLifespan:** has the same value for >90% of the datasaet. We suspect that this category is simply all accounts that have a lifespan of 12+ months, regardless of how much above 12 months they are in age. We have decided not to use this variable due to it being essentially the same value for almost all rows.

- **itemBuyFrequency:** This doesn't seem to be a univariate index. While we know it's a value between 0 and 1, even by removing rows we would not expect this type of skewed distribution favoring the maximal index point. We would expect it to look more like the **webUsage Distribution** chart.

**Bivariate Analysis**

Analyzing the relationship between two variables using pairplots and correlation heatmap.

- Using a pairplot for our numerical variables, we can get to know more about how our variables are related to each other and if we can find patterns. Even though the outliers make it difficult, we find some interesting patterns, 
- With the correlation heatmap, we can analyze the correlation between our variables. When facing a clustering problem, it is important to understand the correlation as it can influence the formation of clusters for some algorithms (e.g., KMeans). A high correlation suggests a linear relationship. In our case, most of the variables have a relatively low correlation and it is therefore not a big concern for this project.

**EDA Conclusion**

After our extensive EDA, we decided to create clusters based on the features **MonthlyPaid** and **avgItemCost**. There are several reasons for this. These varaibles give us information about both how much money the customer spends on average and what type of price range the customer buys from. Even though the average item cost does not give us all the information about the actual price of all items bought by the customer, it acts as a good proxy for that.  
We believe that this segmentation approach to identify clusters where ShopEasy can offer advertising of relevant price ranges to different segments and offer different discounts depending on how much the customer spends, as two examples.

#### Preprocessing




### Experimental Design



### Results



### Conclusions