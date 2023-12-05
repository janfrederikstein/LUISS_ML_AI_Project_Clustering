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




### Experimental Design



### Results



### Conclusions