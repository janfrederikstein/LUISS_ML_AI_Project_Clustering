# Artificial Intelligence & Machine Learning Project

### AI &amp; Machine Learning group project at LUISS Fall 2023

##### Philip Fabrelius & Jan Stein

### 1. Introduction
 
The project aims to help the leading e-commerce platform ShopEasy to provide personalized user experiences, special promotions and improved services to their customers by understanding the buying habits and behaviors of their customers. This is done by applying segmentation to an extensive dataset provided by the platform.

To successfully carry out the task at hand, the following steps were performed:  
- Exploratory Data Analysis (EDA)
- Preprocessing of data
- Testing different clustering models and tuning
- Comparison and evaluation between different clustering models
Supervised classification model for analysis of common attributes within clusters
- Description of results

We identified two key features for our segmentation:  
- `Monthly Paid` - average amount spent on ShopEasy per month
- `Average Item Cost` - average price of items purchased by the customer

Our best clustering model returned the following five segments:
1. `Budget-Conscious` (bottom lower left corner)
2. `Regular Buyers` (above budget-conscious on the y-axis)
3. `Mid-Range Shoppers` (middle on the x-axis, ~0.0-1.0)
4. `Occasional Splurgers` (bottom right on the x-axis)
5. `Big Spenders` (top left on the y-axis)

### 2. Methods

#### 2.1 Imported Libraries
- `Pandas` - powerful and flexible library for data manipulation, analysis and working with dataFrames.
- `Seaborn` - great for statistical graphics.
- `Numpy` - library for large arrays and.matrices and mathematical operations for these.
- `Matplotlib` - great for data visualizations
- `Scikit learn` - great library for preprocessing and machine learning algorithms.

    - StandardScaler
    - KMeans
    - Silhouette_score
    - DBSCAN
    - AgglomerativeClustering
    - GaussianMixture

- `Scipy` - for calculation and visualization of hierarchical clustering.
- `TensorFlow` - for creating artificial neural network.

#### 2.2 EDA

**Description of the Dataset**  
We assume all costs and other features related to money presented as number of dollars.

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

Therefore, and because of the size of the dataset, we have chosen to eliminate features that are not relevant for this investigation.

**Columns to be cut**

1. **paymentCompletionRate:** because it is a payment option related feature
2. **maxSpendLimit:** because it is already represented through spending, which is represented in monthly paid
3. **emergencyCount:** because it just reflects the customers preferred payment option, which is irrelevant for this investigation
4. **emergencyUseFrequency:** because of the same reason
5. **emergencyFunds:** because of the same reason
6. **singleItemCosts:** because they are only related to payment option
7. **MultipleItemCosts:** because they are only related to payment option
8. **singleItemBuyFrequency:** because they are only related to payment option
9. **multipleItemBuyFrequency:** because they are only related to payment option
10. **personId:** This feature is just an internal customer ID and does not matter for our purposes

**Descriptive Statistics**

We used the following methods to learn more about the data:

- `Describe() method` - to get more information about range of values, mean, standard deviatio, number of observations and distribution.
- `Info() method` - to get a concise summary of the DataFrame. This method is useful for quickly understanding the structure of the dataset. Using info(), you can quickly assess which columns may require type conversion or additional preprocessing due to null values or incorrect data types.
- `IsNull().sum() method` - The isnull().sum() method is a two-step operation specifically geared towards identifying missing values in the DataFrame. This method is particularly useful for data cleaning and preprocessing, as handling missing values is a critical step in preparing data for analysis or modeling.

**Data Removal**

From our base set of 8 950 rows, we identified some columns (`AccountTotal, ItemCost, ItemCount, MonthlyPaid`) with minimum values of 0 where we believe that the value should be > 0 to be relevant data points. We therefore decided to drop these rows, 2 258 dropped rows.  
32 missing values were found for leastAmountPaid and they were also dropped. 6 660 rows remain, 74.4% of the original data.

**Feature Engineering**

The dataset includes features for total amount of money spent (accountTotal), total amount of items purchased (itemCount), total cost of those items (itemCosts) and an average monthly spending feature (monthlyPaid). It does however not include a feature for the average item cost.

We add a feature for average item cost (`avgItemCost`) by dividing the itemCosts by itemCount.

**Analyzing Outliers**

We analyzed outliers in the dataset using a `boxplot`. Most columns have many distant outliers except the frequency features created by ShopEasy, which make sense as they are a relative range feature between 0-1 or 0-100. Valuable information for later steps.

Numerical outlier boxplot
![Alt text](images/outlier_boxplot.png)

**Analyzing Categorical Features**

Both categorical features consists of three evenly distributed values. `Location` (New York, Los Angeles and Chicago) and `AccountType` (Premium, Student and Standard).

Categorical countplot
![Alt text](images/categorical_countplot.png)

**Analyzing Numerical Features**

As a part of the univariate analysis with the main purpose of describing and finding patterns for one feature, we used histograms to analyze the distribution of all remaining numerical features, here are some findings:

- `accountLifespan:` has the same value for >90% of the dataset. We suspect that this category is simply all accounts that have a lifespan of 12+ months, regardless of how much above 12 months they are in age. We have decided not to use this feature due to it being essentially the same value for almost all rows.

- `itemBuyFrequency:` This doesn't seem to be a univariate index. While we know it's a value between 0 and 1, even by removing rows we would not expect this type of skewed distribution favoring the maximal index point. We would expect it to look more like the `webUsage Distribution` chart.

Univariate histplot
![Alt text](images/univariate_histplot.png)

**Bivariate Analysis**

Analyzing the relationship between two features using pairplots and correlation heatmap.

- Using a `pairplot` for our numerical features, we can get to know more about how our features are related to each other and if we can find patterns. Even though the outliers make it difficult, we find some interesting patterns. There are also some expected patterns where we for example see a positive relationship between item count and item cost, which makes much sense as purchasing more items often costs more money than purchasing less items.


Pairplot
![Alt text](images/pairplots.png)

- With the `correlation heatmap`, we can analyze the correlation between our features. When facing a clustering problem, it is important to understand the correlation as it can influence the formation of clusters for some algorithms (e.g., KMeans). A high correlation suggests a linear relationship. In our case, most of the features have a relatively low correlation and it is therefore not a big concern for this project.

Correlation heatmap
![Alt text](images/corr_heatmap.png)

**EDA Conclusion**

After our extensive EDA, we decided to create clusters based on the features `MonthlyPaid` and `avgItemCost`. There are several reasons for this. These features give us information about both how much money the customer spends on average and what type of price range the customer buys from. Even though the average item cost does not give us all the information about the actual price of all items bought by the customer, it acts as a good proxy for that.  
We believe that this segmentation approach to identify clusters where ShopEasy can offer advertising of relevant price ranges to different segments and offer different discounts depending on how much the customer spends, as two examples.

#### 2.3 Preprocessing

**Encode Categorical features**

We used the `get_dummies` from pandas on our categorical features `location` and `accountType`. 

**Scale numerical values**

`StandardScaler` is a preprocessing technique used in machine learning and statistics. It standardizes the features of a dataset by removing the mean and dividing by the standard deviation, thus scaling each feature to unit variance. This process can be crucial for the performance of many algorithms, especially those that are sensitive to the scale of features.  
It is important to scale the data to ensure equalizing the importance of features, dealing with different units of measurement enhance interpretability.

**Type of problem & concerns with data distribution**

To analyze the buying behavior, we are specifically interested in the features **'monthlyPaid'** and **'avgItemCost'**. However, these include extreme outliers, as can be seen both in the outlier plots earlier in our EDA, as well as in the difference between the 75% and max columns in the .describe() method used above.

In order to better analyze the relationship of the larger customer group, we removed customers where these values are above 3 standard deviations in either or both of these values. Customers who end up in these extreme localities will instead be offered special services from a Customer Relations manager on an individual basis because it is difficult to generalize from outliers and their existence could seriously hamper the functionality of algorithms used to analyze the dataset. We removed 221 customers (3.3%)

We are left with a much more manageable dataset that is starting to show some patterns. Just from looking at it, we can see that while the majority of customers are centered around the mean 0 (as expected), there seems to be a considerable distinction between customers who have high monthly spending because of expensive items, and customers who buy expensive items but a lot less frequently. The distribution of remaining data points can be seen in this plot.

Scatterplot
![Alt text](images/2variable_distribution_scatterplot.png)

#### 2.4 Clustering Algorithms

We decided to try the following four clustering methods:

- KMeans++
- Hierarchical clustering
- DBSCAN
- Gaussian Mixture

They vary in input parameters, scalability, usecases and distance metric used. Our goal was to find the best clustering method and hyperparameter tuning for our dataset and objective. We provide more details of every method below.

**KMeans++**

Kmeans clusters data by trying to separate samples in groups of equal variances, minimizing the `inertia` which is the average distance between all points of a cluster and its centroid, summed for all clusters. The basic version of KMeans is highly dependent on the random initialization of centroids, which may lead to the model returning a local minimum. The ++ method solves the problem by initializing the centroids far from each other.  
The method requires the user to decide the number of clusters as input before running the model. A helpful method to decide the best number of n is the `the elbow method`. By calculating the inertia for every number of n within a range and plotting the result in a line chart, you can find the point ('the elbow') where the model no longer returns a significantly lower inertia with one extra cluster.

In our model however, the elbow was not as clear as we would prefer. We decided to test another metric, the `Silhouette Score`. It measures how similar an object is to its own cluster and compared to other clusters. Ranging from -1 to +1, a score close to +1 indicates that points are well clustered, a score close to 0 indicates that the clusters are overlapping and a score close to -1 indicates that data points have been assigned to the wrong cluster.  
While the silhouette score might show a higher value for a certain number of clusters, it does not automatically mean that there is any semantic meaning in the clustering or whether the clustering reflects any relevant aspect for our data and objective. In clustering, there is not always an objective best clustering solution and in this case, we combined inertia, silhouette score and our own subjective visual inspection to decide that five clusters was the best solution for our case.

'Elbow method'
![Alt text](images/kmeans_elbow.png)

Cluster subplots
![Alt text](images/kmeans_subplots.png)

Silhouette score
![Alt text](images/kmeans_silhouette.png)

Best cluster
![Alt text](images/kmeans_cluster_winner.png)

**DBSCAN**

This method does not require the user to set the number of clusters *a priori* and no random seeds are involved. DBSCAN works with every shape, it does not assume a convex shape as KMeans for example. Moreover, it handles outliers well and excludes points not belonging to a nearby cluster. The central concept of this method is the `Core sample`, which are samples in areas of high density.   
DBSCAN uses two key inputs, the `minimum number of samples` and `eps` which defines how many points required to be within a certain distance for qualifying as being a core sample.

We tried combinations of eps between 0.05-0.15 and n_samples of 10-30. Higher eps creates larger dense clusters and some smaller more 'random' clusters while larger n_samples generates larger and fewer clusters, therefore we stayed within these ranges. The output for the lower number of samples created several small clusters that don't make much sense while the larger number mainly creates few large clusters without clear features of the segment. We decided to continue with an eps of 0.15 and n_samples of 20.

Cluster subplots
![Alt text](images/dbscan_subplots.png)

Silhouette score
![Alt text](images/dbscan_silhouette.png)

**Hierarchical Clustering**

This method uses a bottom-up approach where each observations starts in its own cluster and then the closest clusters merge, one step at a time until only one clusters remain. There are several possible distance metrics, we decided to use `Ward` which is similar to KMeans for trying to minimize the variance within all clusters but with an agglomerative hierarchical approach.  
The best number of clusters can be analyzed using a `Dendogram` for a tree representation where the vertical distance represents what is called the `Linkage`. Where the linkage is the largest, is where the method merges the two furthest apart clusters.  

In our case, the highest linkage is between two and three clusters, indicating that two clusters are a good fit. However, similar to KMeans, it does not really make any semantic sense and we should consider other options. Looking at the visual representation, five clusters is another good option and we will continue with the two and five cluster options to the next step, but first we will have a look at the next method.

Dendogram
![Alt text](images/hierarchical_dendogram.png)

Linkage for last steps
![Alt text](images/hierarchical_dendogram_steps.png)

Cluster subplots
![Alt text](images/hierarchical_subplots.png)

Silhouette score
![Alt text](images/hierarchical_silhouette.png)

**Gaussian Mixture**

The final tested method was the Gaussian Mixture. This method clusters based on a Gaussian distribution of, in our case two, included features. The clustering results from this method reveals some problems with this method in combination with our dense dataset. We can see it for five clusters where one cluster for the same value on the y-axis is both on the left and right side of two other clusters, and the silhouette score supports our observation. However, we keep the results for two and five clusters for comparison between methods in the next section.

Cluster subplots
![Alt text](images/gaussian_subplots.png)

Silhouette score
![Alt text](images/gaussian_silhouette.png)


#### 2.5 Environment Description

To find a full description of the environment used, please see the appendix at the end of this file.


### 3. Experimental Design

#### 3.1 Comparison between methods

To decide which method that is best for our data and objective we compare the best clustering from each method. We have decided to disregard the time it takes to run the models as a performance metric, as all of the clustering methods used have negligible run times that do not affect our choice.

Comparing and evaluating clustering methods is a difficult task to perform, especially when there is no ground truth to compare the results to. If we had, we could as well use a supervised classification algorithm to predict the value. As we have discussed before, there are metrics available to evaluate clustering outputs, like silhouette score. However, these often don't work well in practice where shapes might be complex. Even if our silhouette score indicates a well-formed cluster, it might lack semantic meaning and reflection of aspects that are interesting for us. This leaves room for our subjective opinion looking at the visual representation, this is possible as we only use two features. 

Cluster subplots
![Alt text](images/cluster_comparison_subplots.png)

Silhouette score comparison
![Alt text](images/silhouette_comparison.png)

Despite high silhouette scores, we have decided to disregard segmentations with fewer than 3 clusters as we don't recognize any semantic meaning in these and they don't reflect our intention of properly creating actionable customer segments that can be used targeted advertisement and marketing campaigns. This eliminates:
- DBSCAN
- Gaussian Mixture with n=2
- Agglomerative/Hierarchical Clustering with n=2

leaving us with 3 remaining options:
- KMeans
- Agglomerative/Hierarchical Clustering with n=5
- Gaussian Mixture with n=5

We see issues with Gaussian Mixture n=5 both when it comes to the split cluster around (-0.5,0), and because of the vertical size of the right cluster, stretching from very low to very high monthly paid, essentially not discriminating between largely different customers.

Comparing the last 2 options, we prefer KMeans as it has a more semantically logical divide between spending habits and item cost, as well as a more diagonal split between high spending and high item cost customers.

#### 3.2 Supervised Model to Predict Segment

To get a better understanding for the properties and attributes of the segments created in our clustering methods, we do some brief investigation with classification algorithms to determine if there are any common denominators apart from the buying behaviour of our segments.

We add a new column to the already scaled and processed dataset that reflects the assigned cluster with an integer value (0-4). The features that were used to create the clustering are removed to not leak information into the model, otherwise we would end up with a perfect prediction.

We use two different classification models:
- Decision Tree
- Artificial Neural Network

The primary objective of the decision tree model is to assist with determining feature importance due to the inherently high interpretability of decision trees. A GridSearchCV is subsequently used to improve model performance through tuning of hyperparameters.

An ANN is subsequently used in an attempt to maximize model performance on the available data. Various setups (2-8 hidden layers with 8-50 neurons per layer, 20-200 epochs, relu/sigmoid activation) were investigated in an attempt to minimize overfitting whilst maximizing generalization and accuracy on the test set.

### 4. Results

#### 4.1 Main Result from Clustering

Our final result from our KMeans++ method is illustrated in `figure X` and includes the following five clusters:

1. Budget-Conscious (bottom lower left corner)
2. Regular Buyers (above budget-conscious on the y-axis)
3. Mid-Range Shoppers (middle on the x-axis, ~0.0-1.0)
4. Occasional Splurgers (bottom right on the x-axis)
5. Big Spenders (top left on the y-axis)

The budget-conscious cluster contains customers who spend relatively small amount of money monthly and buys relatively cheap items on average. These are not the most profitable customers and might not require much special attention. However, they are many and it is good to have a good base offer to maintain the revenue stream from these customers.

The regular buyers spend more money than the budget-conscious segment, but still buy relatively cheap items. For this segment, we should put more focus on relevant product offerings and content to make sure that they continue buying products.

The mid-range shoppers buy slightly more expensive products on average but don't spend as much in total as the regular buyers. Here it becomes more important with offering the right products to this group, as they buy fewer but more expensive items.

The occasional splurgers can be a difficult group to provide with relevant content. They rarely buy items but when they do, they are buying expensive items. For this group, the content should focus on premium products.

The last group, the big spenders, tend to buy a lot of products within the relatively cheap price range but there can also be cases where they buy some expensive products and many cheaper products. We know that they have money to spend and we should make our biggest effort for this group.

#### 4.2 Main Result from Classification

Feature importance decision tree
![Alt text](images/decision_tree_feature_importance.png)

While a glimpse at the feature importance graph above gives the appearance of some insights, it would be a stretch to call the results from our classification models anything less than severely disappointing. Even after tuning, the decision tree approach barely cracked 60% accuracy, not significantly better than flipping a coin. And even our attempts to improve performance with a less interpretable but more powerful model like ANN was nothing less than a failure. Additionally, since our budget-conscious cluster holds approximately half of the customers with around average or less buying behaviour, the models barely cracked the stupidest possible approach of taking the average.

It seems that whilst the clusters quite clearly separate customers when it comes to buying behaviour, the attributes outside of this category overlap too much to make it possible to predict the appropriate clusters on them (at least with the information used in our model). 

This in itself however is an interesting finding, as it suggests that similar buying behaviour can arise in vastly different customer patterns. In order to better describe the customers in the clusters, one would likely need access to more informative attributes about demographics.


### 5. Conclusions

The main take-away from this investigation is that it is possible to properly cluster customers into segments based on their buying behaviour using the average cost of the items that they buy in combination with the average amount of money that they spend per month. ShopEasy can use the 5 identified clusters to tailor their campaigns and marketing efforts appropriately to both the value of a customers spending, as well as by catering towards the typical price range of items purchased by the customers. These clusters can also prove valuable when deciding which type of content to show to customers in webshop settings by increasing the likelihood of customers being shown items and promotions that they are actually interested in buying, thus representing a value proposition for the company.

What has become abundantly clear from our classification efforts, is that there seems to be a large overlap between segments when it comes to other factors, not directly related to the identified features for spending habits. In essence, while customers can be divided into clusters after making purchases, the company might be interested in predicting early which type of cluster a customer will eventually belong to, to maximize the lifetime value potential from them. This will however require additional information about the customers, potentially more demographic features could be promising, if they can be obtained. Also, whilst our investigation discusses the average cost of items, it does not cover topics such as product categories, which could be way more influential in customer buying behaviour. A good next step would probably be to dig deeper into the exact items purchased by customers and trying to segment on these type of categories, potentially in combination with the value-oriented segments we have identified. It is essental to understand, that a customers purchasing choices are not only determined by the price of an item, but rather by the customers actual interest in the item. This dimension is omitted entirely from this dataset, and could be one of the leading causes why predicting cluster identify is difficult.




## Appendix: Environment Description

# Name                    Version                   Build  Channel
_tflow_select             2.2.0                     eigen  
abseil-cpp                20211102.0           hc377ac9_0  
absl-py                   1.4.0           py311hca03da5_0  
aiofiles                  22.1.0          py311hca03da5_0  
aiohttp                   3.8.5           py311h80987f9_0  
aiosignal                 1.2.0              pyhd3eb1b0_0  
aiosqlite                 0.18.0          py311hca03da5_0  
anyio                     3.5.0           py311hca03da5_0  
appnope                   0.1.2           py311hca03da5_1001  
argon2-cffi               21.3.0             pyhd3eb1b0_0  
argon2-cffi-bindings      21.2.0          py311h80987f9_0  
asttokens                 2.0.5              pyhd3eb1b0_0  
astunparse                1.6.3                      py_0  
async-timeout             4.0.2           py311hca03da5_0  
attrs                     23.1.0          py311hca03da5_0  
babel                     2.11.0          py311hca03da5_0  
backcall                  0.2.0              pyhd3eb1b0_0  
beautifulsoup4            4.12.2          py311hca03da5_0  
blas                      1.0                    openblas  
bleach                    4.1.0              pyhd3eb1b0_0  
blinker                   1.6.2           py311hca03da5_0  
boost-cpp                 1.73.0              h1a28f6b_12  
bottleneck                1.3.5           py311ha0d4635_0  
brotli                    1.0.9                h1a28f6b_7  
brotli-bin                1.0.9                h1a28f6b_7  
brotlipy                  0.7.0           py311h80987f9_1002  
bzip2                     1.0.8                h620ffc9_4  
c-ares                    1.19.1               h80987f9_0  
ca-certificates           2023.08.22           hca03da5_0  
cachetools                4.2.2              pyhd3eb1b0_0  
cairo                     1.16.0               h302bd0f_5  
catalogue                 2.0.7           py311hca03da5_0  
certifi                   2023.11.17      py311hca03da5_0  
cffi                      1.15.1          py311h80987f9_3  
charset-normalizer        2.0.4              pyhd3eb1b0_0  
click                     8.0.4           py311hca03da5_0  
colorama                  0.4.6           py311hca03da5_0  
comm                      0.1.2           py311hca03da5_0  
confection                0.0.4           py311hb6e6a13_0  
contourpy                 1.0.5           py311h48ca7d4_0  
cryptography              41.0.3          py311h3c57c4d_0  
cycler                    0.11.0             pyhd3eb1b0_0  
cymem                     2.0.6           py311h313beb8_0  
cyrus-sasl                2.1.28               h458e800_1  
cython-blis               0.7.9           py311hb9f6ed7_0  
debugpy                   1.6.7           py311h313beb8_0  
decorator                 5.1.1              pyhd3eb1b0_0  
defusedxml                0.7.1              pyhd3eb1b0_0  
dnspython                 2.4.2                    pypi_0    pypi
en-core-web-sm            3.5.0                    pypi_0    pypi
entrypoints               0.4             py311hca03da5_0  
executing                 0.8.3              pyhd3eb1b0_0  
expat                     2.5.0                h313beb8_0  
flatbuffers               2.0.0                hc377ac9_0  
font-ttf-dejavu-sans-mono 2.37                 hd3eb1b0_0  
font-ttf-inconsolata      2.001                hcb22688_0  
font-ttf-source-code-pro  2.030                hd3eb1b0_0  
font-ttf-ubuntu           0.83                 h8b1ccd4_0  
fontconfig                2.14.1               hee714a5_2  
fonts-anaconda            1                    h8fa9717_0  
fonts-conda-ecosystem     1                    hd3eb1b0_0  
fonttools                 4.25.0             pyhd3eb1b0_0  
freetype                  2.12.1               h1192e45_0  
fribidi                   1.0.10               h1a28f6b_0  
frozenlist                1.3.3           py311h80987f9_0  
gast                      0.4.0              pyhd3eb1b0_0  
gdk-pixbuf                2.42.10              h80987f9_0  
gettext                   0.21.0               h13f89a0_1  
giflib                    5.2.1                h80987f9_3  
glib                      2.69.1               h514c7bf_2  
google-auth               2.22.0          py311hca03da5_0  
google-auth-oauthlib      0.5.2           py311hca03da5_0  
google-pasta              0.2.0              pyhd3eb1b0_0  
graphite2                 1.3.14               hc377ac9_1  
graphviz                  2.50.0               hf331ead_1  
grpc-cpp                  1.48.2               h877324c_0  
grpcio                    1.48.2          py311h877324c_0  
gst-plugins-base          1.14.1               h313beb8_1  
gstreamer                 1.14.1               h80987f9_1  
gts                       0.7.6                hde733a8_3  
h5py                      3.9.0           py311hba6ad2f_0  
harfbuzz                  4.3.0                he9eebac_1  
hdf5                      1.12.1               h160e8cb_2  
icu                       68.1                 hc377ac9_0  
idna                      3.4             py311hca03da5_0  
imageio                   2.31.4          py311hca03da5_0  
ipykernel                 6.25.0          py311hb6e6a13_0  
ipython                   8.15.0          py311hca03da5_0  
ipython_genutils          0.2.0              pyhd3eb1b0_1  
ipywidgets                8.0.4           py311hca03da5_0  
jax                       0.4.21                   pypi_0    pypi
jedi                      0.18.1          py311hca03da5_1  
jinja2                    3.1.2           py311hca03da5_0  
joblib                    1.2.0           py311hca03da5_0  
jpeg                      9e                   h80987f9_1  
json5                     0.9.6              pyhd3eb1b0_0  
jsonschema                4.17.3          py311hca03da5_0  
jupyter                   1.0.0           py311hca03da5_8  
jupyter_client            7.4.9           py311hca03da5_0  
jupyter_console           6.6.3           py311hca03da5_0  
jupyter_core              5.3.0           py311hca03da5_0  
jupyter_events            0.6.3           py311hca03da5_0  
jupyter_server            1.23.4          py311hca03da5_0  
jupyter_server_fileid     0.9.0           py311hca03da5_0  
jupyter_server_ydoc       0.8.0           py311hca03da5_1  
jupyter_ydoc              0.2.4           py311hca03da5_0  
jupyterlab                3.6.3           py311hca03da5_0  
jupyterlab_pygments       0.1.2                      py_0  
jupyterlab_server         2.22.0          py311hca03da5_0  
jupyterlab_widgets        3.0.5           py311hca03da5_0  
keras                     2.12.0          py311hca03da5_0  
keras-preprocessing       1.1.2              pyhd3eb1b0_0  
kiwisolver                1.4.4           py311h313beb8_0  
krb5                      1.20.1               h8380606_1  
langcodes                 3.3.0              pyhd3eb1b0_0  
lcms2                     2.12                 hba8e193_0  
lerc                      3.0                  hc377ac9_0  
libboost                  1.73.0              h49e8a49_12  
libbrotlicommon           1.0.9                h1a28f6b_7  
libbrotlidec              1.0.9                h1a28f6b_7  
libbrotlienc              1.0.9                h1a28f6b_7  
libclang                  16.0.6                   pypi_0    pypi
libclang13                14.0.6          default_h24352ff_1  
libcurl                   8.2.1                h0f1d93c_0  
libcxx                    14.0.6               h848a8c0_0  
libdeflate                1.17                 h80987f9_1  
libedit                   3.1.20221030         h80987f9_0  
libev                     4.33                 h1a28f6b_1  
libffi                    3.4.4                hca03da5_0  
libgd                     2.3.3                h14f8a72_1  
libgfortran               5.0.0           11_3_0_hca03da5_28  
libgfortran5              11.3.0              h009349e_28  
libiconv                  1.16                 h1a28f6b_2  
libllvm14                 14.0.6               h7ec7a93_3  
libnghttp2                1.52.0               h10c0552_1  
libopenblas               0.3.21               h269037a_0  
libpng                    1.6.39               h80987f9_0  
libpq                     12.15                h449679c_1  
libprotobuf               3.20.3               h514c7bf_0  
librsvg                   2.54.4               hb3bd4c3_3  
libsodium                 1.0.18               h1a28f6b_0  
libssh2                   1.10.0               h449679c_2  
libtiff                   4.5.1                h313beb8_0  
libtool                   2.4.6             h313beb8_1009  
libwebp                   1.3.2                ha3663a8_0  
libwebp-base              1.3.2                h80987f9_0  
libxml2                   2.10.4               h372ba2a_0  
libxslt                   1.1.37               habca612_0  
llvm-openmp               14.0.6               hc6e5704_0  
lxml                      4.9.3           py311h50ffb84_0  
lz4-c                     1.9.4                h313beb8_0  
markdown                  3.4.1           py311hca03da5_0  
markupsafe                2.1.1           py311h80987f9_0  
matplotlib                3.7.2           py311hca03da5_0  
matplotlib-base           3.7.2           py311h7aedaa7_0  
matplotlib-inline         0.1.6           py311hca03da5_0  
matplotlib-venn           0.11.9                   pypi_0    pypi
mistune                   0.8.4           py311h80987f9_1000  
ml-dtypes                 0.3.1                    pypi_0    pypi
multidict                 6.0.2           py311h80987f9_0  
munkres                   1.1.4                      py_0  
murmurhash                1.0.7           py311h313beb8_0  
mysql                     5.7.24               he1ceea5_2  
mysql-connector-python    8.1.0                    pypi_0    pypi
nbclassic                 0.5.5           py311hca03da5_0  
nbclient                  0.5.13          py311hca03da5_0  
nbconvert                 6.5.4           py311hca03da5_0  
nbformat                  5.9.2           py311hca03da5_0  
ncurses                   6.4                  h313beb8_0  
nest-asyncio              1.5.6           py311hca03da5_0  
nltk                      3.8.1           py311hca03da5_0  
notebook                  6.5.4           py311hca03da5_1  
notebook-shim             0.2.2           py311hca03da5_0  
nspr                      4.35                 h313beb8_0  
nss                       3.89.1               h313beb8_0  
numexpr                   2.8.7           py311h6dc990b_0  
numpy                     1.23.5          py311hb2c0538_0  
numpy-base                1.23.5          py311h9eb1c70_0  
oauthlib                  3.2.2           py311hca03da5_0  
openjpeg                  2.3.0                h7a6adac_2  
openssl                   1.1.1w               h1a28f6b_0  
opt_einsum                3.3.0              pyhd3eb1b0_1  
packaging                 23.1            py311hca03da5_0  
pandas                    2.0.3           py311h7aedaa7_0  
pandocfilters             1.5.0              pyhd3eb1b0_0  
pango                     1.50.7               h7271ec9_0  
parso                     0.8.3              pyhd3eb1b0_0  
pathy                     0.10.1          py311hca03da5_0  
pcre                      8.45                 hc377ac9_0  
pexpect                   4.8.0              pyhd3eb1b0_3  
pickleshare               0.7.5           pyhd3eb1b0_1003  
pillow                    10.0.1          py311h3b245a6_0  
pip                       23.2.1          py311hca03da5_0  
pixman                    0.40.0               h1a28f6b_0  
platformdirs              3.10.0          py311hca03da5_0  
ply                       3.11            py311hca03da5_0  
poppler                   22.12.0              h52f4003_3  
poppler-data              0.4.11               hca03da5_1  
preshed                   3.0.6           py311h313beb8_0  
prometheus_client         0.14.1          py311hca03da5_0  
prompt-toolkit            3.0.36          py311hca03da5_0  
prompt_toolkit            3.0.36               hd3eb1b0_0  
protobuf                  4.21.12                  pypi_0    pypi
psutil                    5.9.0           py311h80987f9_0  
ptyprocess                0.7.0              pyhd3eb1b0_2  
pure_eval                 0.2.2              pyhd3eb1b0_0  
pyasn1                    0.4.8              pyhd3eb1b0_0  
pyasn1-modules            0.2.8                      py_0  
pycparser                 2.21               pyhd3eb1b0_0  
pydantic                  1.10.12         py311h80987f9_1  
pydot                     1.4.2                    pypi_0    pypi
pygments                  2.15.1          py311hca03da5_1  
pyjwt                     2.4.0           py311hca03da5_0  
pymongo                   4.6.1                    pypi_0    pypi
pyopenssl                 23.2.0          py311hca03da5_0  
pyparsing                 3.0.9           py311hca03da5_0  
pyqt                      5.15.7          py311h313beb8_0  
pyqt5-sip                 12.11.0         py311h313beb8_0  
pyrsistent                0.18.0          py311h80987f9_0  
pysocks                   1.7.1           py311hca03da5_0  
python                    3.11.5               hc0d8a6c_0  
python-dateutil           2.8.2              pyhd3eb1b0_0  
python-fastjsonschema     2.16.2          py311hca03da5_0  
python-flatbuffers        2.0                pyhd3eb1b0_0  
python-graphviz           0.20.1          py311hca03da5_0  
python-json-logger        2.0.7           py311hca03da5_0  
python-tzdata             2023.3             pyhd3eb1b0_0  
pytz                      2023.3.post1    py311hca03da5_0  
pyyaml                    6.0             py311h80987f9_1  
pyzmq                     23.2.0          py311h313beb8_0  
qt-main                   5.15.2               h8c8f68c_9  
qt-webengine              5.15.9               h2903aaf_7  
qtconsole                 5.4.2           py311hca03da5_0  
qtpy                      2.2.0           py311hca03da5_0  
qtwebkit                  5.212                h19f419d_5  
re2                       2022.04.01           hc377ac9_0  
readline                  8.2                  h1a28f6b_0  
regex                     2023.10.3       py311h80987f9_0  
requests                  2.31.0          py311hca03da5_0  
requests-oauthlib         1.3.0                      py_0  
rfc3339-validator         0.1.4           py311hca03da5_0  
rfc3986-validator         0.1.1           py311hca03da5_0  
rsa                       4.7.2              pyhd3eb1b0_1  
scikit-learn              1.3.2                    pypi_0    pypi
scipy                     1.11.3          py311hc76d9b0_0  
seaborn                   0.12.2          py311hca03da5_0  
send2trash                1.8.0              pyhd3eb1b0_1  
setuptools                68.0.0          py311hca03da5_0  
shellingham               1.5.0           py311hca03da5_0  
sip                       6.6.2           py311h313beb8_0  
six                       1.16.0             pyhd3eb1b0_1  
smart_open                5.2.1           py311hca03da5_0  
snappy                    1.1.9                hc377ac9_0  
sniffio                   1.2.0           py311hca03da5_1  
soupsieve                 2.5             py311hca03da5_0  
spacy                     3.5.3           py311h30ceab6_0  
spacy-legacy              3.0.12          py311hca03da5_0  
spacy-loggers             1.0.4           py311hca03da5_0  
sqlite                    3.41.2               h80987f9_0  
srsly                     2.4.6           py311h313beb8_0  
stack_data                0.2.0              pyhd3eb1b0_0  
tensorboard               2.12.1          py311hca03da5_0  
tensorboard-data-server   0.7.0           py311ha6e5c4f_0  
tensorboard-plugin-wit    1.8.1           py311hca03da5_0  
tensorflow                2.12.0          eigen_py311h689d69b_0  
tensorflow-base           2.12.0          eigen_py311h0a52ebb_0  
tensorflow-estimator      2.12.0          py311hca03da5_0  
termcolor                 2.1.0           py311hca03da5_0  
terminado                 0.17.1          py311hca03da5_0  
thinc                     8.1.10          py311h30ceab6_0  
threadpoolctl             2.2.0              pyh0d69192_0  
tinycss2                  1.2.1           py311hca03da5_0  
tk                        8.6.12               hb8d0fd4_0  
toml                      0.10.2             pyhd3eb1b0_0  
tornado                   6.3.3           py311h80987f9_0  
tqdm                      4.65.0          py311hb6e6a13_0  
traitlets                 5.7.1           py311hca03da5_0  
typer                     0.4.1           py311hca03da5_0  
typing-extensions         4.7.1           py311hca03da5_0  
typing_extensions         4.7.1           py311hca03da5_0  
tzdata                    2023c                h04d1e81_0  
urllib3                   1.26.16         py311hca03da5_0  
wasabi                    0.9.1           py311hca03da5_0  
wcwidth                   0.2.5              pyhd3eb1b0_0  
webencodings              0.5.1           py311hca03da5_1  
websocket-client          0.58.0          py311hca03da5_4  
werkzeug                  2.2.3           py311hca03da5_0  
wheel                     0.35.1             pyhd3eb1b0_0  
widgetsnbextension        4.0.5           py311hca03da5_0  
wrapt                     1.14.1          py311h80987f9_0  
xz                        5.4.2                h80987f9_0  
y-py                      0.5.9           py311ha6e5c4f_0  
yaml                      0.2.5                h1a28f6b_0  
yarl                      1.8.1           py311h80987f9_0  
ypy-websocket             0.8.2           py311hca03da5_0  
zeromq                    4.3.4                hc377ac9_0  
zlib                      1.2.13               h5a0b063_0  
zstd                      1.5.5                hd90d995_0  