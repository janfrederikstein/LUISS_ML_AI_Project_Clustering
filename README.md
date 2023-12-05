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

- Using a pairplot for our numerical variables, we can get to know more about how our variables are related to each other and if we can find patterns. Even though the outliers make it difficult, we find some interesting patterns. There are also some expected patterns where we for example see a positive relationship between item count and item cost, which makes much sense as purchasing more items often costs more money than purchasing less items.
- With the correlation heatmap, we can analyze the correlation between our variables. When facing a clustering problem, it is important to understand the correlation as it can influence the formation of clusters for some algorithms (e.g., KMeans). A high correlation suggests a linear relationship. In our case, most of the variables have a relatively low correlation and it is therefore not a big concern for this project.

**EDA Conclusion**

After our extensive EDA, we decided to create clusters based on the features **MonthlyPaid** and **avgItemCost**. There are several reasons for this. These varaibles give us information about both how much money the customer spends on average and what type of price range the customer buys from. Even though the average item cost does not give us all the information about the actual price of all items bought by the customer, it acts as a good proxy for that.  
We believe that this segmentation approach to identify clusters where ShopEasy can offer advertising of relevant price ranges to different segments and offer different discounts depending on how much the customer spends, as two examples.

#### Preprocessing

**Encode Categorical Variables**

We used the `get_dummies` from pandas on our categorical features **location** and **accountType**. 

**Scale numerical values**

`StandardScaler` is a preprocessing technique used in machine learning and statistics. It standardizes the features of a dataset by removing the mean and scaling each feature to unit variance. This process can be crucial for the performance of many algorithms, especially those that are sensitive to the scale of features.  
It is important to scale the data to ensure equalizing the importance of features, dealing with different units of measurement enhance interpretability.

**Type of problem & concerns with data distribution**

To analyse the buying behaviour, we are specifically interested in the variables **'monthlyPaid'** and **'avgItemCost'**. However, these include extreme outliers, as can be seen both in the outlier plots earlier in our EDA, as well as in the difference between the 75% and max columns in the .describe() method used above.

In order to better analyse the relationship of the larger customer group, we removed customers where these values are above 3 standard deviations in either or both of these values. Customers who end up in these extreme localities will instead be offered special services from a Customer Relations manager on an individual basis because it is difficult to generalize from outliers and their existence could seriously hamper the functionality of algorithms used to analyse the dataset. We removed 221 customers (3.3%)

We are left with a much more managable dataset that is starting to show some patterns. Just from looking at it, we can see that while the majority of customers are centered around the mean 0 (as expected), there seems to be a considerable distinction between customers who have high monthly spending because of expensive items, and customers who buy expensive items but a lot less frequently. The distribution of remaining data points can be seen in `plot x`.

#### Clustering Algorithms

We decided to try the following four clustering methods:

- KMeans++
- Hierarchical clustering
- DBSCAN
- Gaussian Mixture

They vary in input parameters, scalability, usecases and distance metric used. Our goal was to find the best clustering method and hyperparameter tuning for our dataset and objective. We provide more details of every method below.

**KMeans++**

Kmeans clusters data by trying to separate samples in groups of equal variance, minimizing the `inertia` which is the average distance between all points of a cluster and its centroid, summed for all clusters. The basic version of KMeans is highly dependent on the random initialization of centroids, which may lead to the model returning a local minimum. The ++ method solves the problem by initializing the centroids far from each other.  
The method requires the user to decide the number of clusters as input before running the model. A helpful method to decide the best number of n is the `the elbow method`. By calculating the inertia for every number of n within a range and plotting the result in a line chart, you can find the point ('the elbow') where the model no longer returns a significantly lower inertia with one extra cluster.

In our model however, the elbow was not as clear as in other datasets. We decided to test another mettric, the `Silhouette Score`. It measures how similar an object is to its own cluster and compared to other clusters. Ranging from -1 to +1, a score close to +1 indicates that points are well clustered, a score close to 0 indicates that the clusters are overlapping and a score close to -1 indicates that data points have been assigned to the wrong cluster.  
While the silhouette score might show a higher value for a certain number of clusters, it does not automatically mean that there is any semantic meaning in the clustering or whether the clustering reflects any relevant aspect for our data and objective. In clustering, there is not always an objective best clustering solution and in this case we combined inertia, silhouette score and our own subjective visual inspection to decide that five clusters was the best solution for our case.

**Hierarchical Clsutering**

This method uses a bottom up approach where each observations starts in its own cluster and then the closest clusters merge, one step at a time until only one clusters remain. There are several possible distance metrics, we decided to use `Ward` which is similar to KMeans for trying to minimize the variance within all clusters but with an agglomerative hierarchical approach.  
The best number of clusters can be analyzed using a `Dendogram` for a tree representation where the vertical distance represents the what is called the `Linkage`. Where the linkage is the largest, is where the method merges the two furthest apart clusters.  

In our case, the highest linkage is between two and three clusters, indicating that two clusters is a good fit. However, similar to KMeans, it does not really make any semantic sense and we should consider other options. Looking at the visual respresentation, five clusters is another good option and we will continue with the two and five cluster options to the next step, but first we will have a look at the next method.

**DBSCAN**

This method does not require the user to set the number of clusters *a priori* and no random seeds are involved. DBSCAN works with every shape, it does not assume a convex shape as KMeans for example. Moreover, it handles outliers well and excludes points not belonging to a nearby cluster. The central concept of this method is the `Core sample`, which are samples in areas of high density.   
DBSCAN uses two key inputs, the `minimum number of samples` and `eps` which defines how many points required to be within a certain distance for qualifying as being a core sample.

We tried combinations of eps between 0.05-0.15 and n_samples of 10-30. The output for the lower number of samples created several small clusters that don't make much sense while the larger number mainly creates few large clusters without clear features of the segment. We decided to continue with an eps of 0.15 and n_samples of 20.

**Gaussian Mixture**

The final method testes was the Gaussian Mixture. This method clusters based on a Gaussian distribution of, in our case two, included features. The clustering results from this method reveals som problems with this method in combination with our dense dataset. We can see it for five clusters where one cluster for the same value on the y-axis is both on the left and right side of two other clusters, and the silhouette score supports our observation. However, we keep the results for two and five clusters for comparison between methods in the next section.


### Experimental Design



### Results



### Conclusions