import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

creditcard_df = pd.read_csv('marketing_data.csv')

print(creditcard_df)

# to print the mean, minimum and maximum
print('Average, Minimum, Maximum = ', creditcard_df['BALANCE'].mean(), creditcard_df['BALANCE'].min(),
      creditcard_df['BALANCE'].max())
print()
print()

# used to get statistical summary of our data
print(creditcard_df.describe())


print()
print()
# Obtain the features (row) of the customer who made the maximum "ONEOFF_PURCHASES"
i = creditcard_df['ONEOFF_PURCHASES'].max()
print(creditcard_df[creditcard_df['ONEOFF_PURCHASES'] == i])


print()
print()
# Obtain the features of the customer who made the maximum cash advance transaction?
# how many cash advance transactions did that customer make? how often did he/she pay their bill?
x = creditcard_df['CASH_ADVANCE'].max()
print(creditcard_df[creditcard_df['CASH_ADVANCE'] == x])

# to check if there is any missing value(s)
sns.heatmap(creditcard_df.isnull(), yticklabels=False, cbar=False, cmap="Blues")


print()
print()
# summary of null values
print(creditcard_df.isnull().sum())


print()
print()
# Fill up the missing elements with mean of the 'MINIMUM_PAYMENT'
creditcard_df.loc[(creditcard_df['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] \
    = creditcard_df['MINIMUM_PAYMENTS'].mean()
print(creditcard_df.isnull().sum())


print()
print()
# Fill up the missing elements with mean of the 'CREDIT_LIMIT'
creditcard_df.loc[(creditcard_df['CREDIT_LIMIT'].isnull() == True), 'CREDIT_LIMIT'] \
    = creditcard_df['CREDIT_LIMIT'].mean()
print(creditcard_df.isnull().sum())


print()
print()
# Let's see if we have duplicated entries in the data
print("Duplicated entries = ", creditcard_df.duplicated().sum())


print()
print()
# dropping the 'CUST_ID' (customer id) column since we didn't need it
creditcard_df.drop('CUST_ID', axis=1, inplace=True)
print(creditcard_df)


print()
print()
# prints the number of columns; just a verification that the column has been dropped
n = len(creditcard_df.columns)
print("Number of columns =", n)


print()
print()
# gives a list of the columns in our dataframe
print(creditcard_df.columns)


# visualisation using plots
plt.figure(figsize=(10, 50))
for i in range(len(creditcard_df.columns)):
    plt.subplot(17, 1, i + 1)
    sns.distplot(creditcard_df[creditcard_df.columns[i]], kde_kws={"color": "b", "lw": 3, "label": "KDE"},
                 hist_kws={"color": "g"})
    plt.title(creditcard_df.columns[i])

plt.tight_layout()


# scaling the data
scaler = StandardScaler()
creditcard_df_scaled = scaler.fit_transform(creditcard_df)

print()
print()
print(creditcard_df_scaled.shape)

print()
print()
print(creditcard_df_scaled)


# Here we use and implement the ELBOW METHOD to find the optimum number of clusters
# that need to be taken for a particular task
# for this example, the answer was 7 or 8 clusters based on the plot acquired
scores = []
values = range(1, 20)

for i in values:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(creditcard_df_scaled)
    scores.append(kmeans.inertia_)

plt.plot(scores, 'bx-')


# the change made in the above code as shown below; will give the optimal number of clusters only
# for the first seven features of our data set instead of considering the entire dataset
# any number can taken, 7 is just taken as an example
# for this example, the answer was around 5 clusters based on the plot acquired

"""
# to check that only the first 7 features are being considered
print(creditcard_df_scaled[:, :7].shape)

scores = []
values = range(1, 20)

for i in values:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(creditcard_df_scaled[:, :7])
    scores.append(kmeans.inertia_)

plt.plot(scores, 'bx-')
"""

# considering 7 clusters
kmeans = KMeans(7)

kmeans.fit(creditcard_df_scaled)
labels = kmeans.labels_

print()
print()
# this line of code will give the centroid of all of the different clusters
print(kmeans.cluster_centers_.shape)


print()
print()
# we create a new data frame to get all the centroids
cluster_centers = pd.DataFrame(data=kmeans.cluster_centers_, columns=[creditcard_df.columns])
print(cluster_centers)


print()
print()
# to understand what these numbers mean, we perform inverse transformation
cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data=cluster_centers, columns=[creditcard_df.columns])
print(cluster_centers)

"""
** IMPORTANT **
the above obtained table / data frame is the most important achievement in our entire analysis
it is based on this that we will cluster of customers into 4 groups 
Data from this table / data frame will read and analysed to tally with the required conditions
and thus form different clusters
"""


"""
                          THE REQUIRED CONDITIONS FOR CLUSTERING
First Customers cluster (Transactors): Those are customers who pay least amount of interest 
charges and careful with their money, Cluster with lowest balance ($104) and cash advance ($303), 
Percentage of full payment = 23%

Second customers cluster (revolvers) who use credit card as a loan (most lucrative sector): 
highest balance ($5000) and cash advance (~$5000), low purchase frequency, high cash advance 
frequency (0.5), high cash advance transactions (16) and low percentage of full payment (3%)

Third customer cluster (VIP/Prime): high credit limit $16K and highest percentage of full payment, 
target for increase credit limit and increase spending habits

Fourth customer cluster (low tenure): these are customers with low tenure (7 years), low balance
"""

print()
print(labels.shape)
print()
print(labels.max())
print()
print(labels.min())


print()
y_kmeans = kmeans.fit_predict(creditcard_df_scaled)
print(y_kmeans)


# concatenate the clusters labels to our original data frame
creditcard_df_cluster = pd.concat([creditcard_df, pd.DataFrame({'cluster':labels})], axis = 1)
creditcard_df_cluster.head()

# Plot the histogram of various clusters
for i in creditcard_df.columns:
    plt.figure(figsize=(35, 5))
    for j in range(7):
        plt.subplot(1, 7, j + 1)
        cluster = creditcard_df_cluster[creditcard_df_cluster['cluster'] == j]
        cluster[i].hist(bins=20)
        plt.title('{}    \nCluster {} '.format(i, j))

    # plt.show()


print()
print()
# now we obtain the principal components
pca = PCA(n_components=2)
principal_component = pca.fit_transform(creditcard_df_scaled)
print(principal_component)


print()
print()
# now create a data frame with the two components
pca_df = pd.DataFrame(data=principal_component, columns=['pca1','pca2'])
print(pca_df.head())

print()
print()
# concatenate the clusters labels to the data frame
pca_df = pd.concat([pca_df, pd.DataFrame({'cluster': labels})], axis=1)
print(pca_df.head())


print()
print()
plt.figure(figsize=(10,10))
ax = sns.scatterplot(x="pca1", y="pca2", hue="cluster", data=pca_df,
                     palette=['red', 'green', 'blue', 'pink', 'yellow', 'gray', 'purple'])
# plt.show()
