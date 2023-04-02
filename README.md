---

# Leveraging Unsupervised Learning Techniques for Data Clustering and Dimensionality Reduction

---

### Introduction
Unsupervised learning is a type of machine learning where the model is not trained on any labeled data. It instead learns the patterns and relationships of the data by itself by using similarities like distances between data points. Unsupervised learning is an important technique in the data science field as it can be used to perform a wide range of tasks such as data clustering, anomaly detection, entity segmentation, and dimensionality reduction.
In this article, we going to focus on data clustering and dimensionality reduction using unsupervised learning techniques. We will use PCA for dimensionality reduction and various clustering to compare results. This blog just give an overview of the whole process which can be found on GitHub using this link;
GitHub …
You can't perform that action at this time. You signed in with another tab or window. You signed out in another tab or…github.com

---

### Data Description
We will be working with a real-world dataset from the UCI Machine Learning Repository. The data can be accessed from here. The dataset contains reviews of travel destinations in East Asia. Each review is mapped on a scale of 0 to 4, where 4 is Excellent and 0 is Terrible. The dataset contains 11 attributes, including the user ID who wrote the review. Our goal is to cluster the reviews based on their similarity and reduce the dimensionality of the dataset for better visualization.
Data Preprocessing
Before we begin with clustering and dimensionality reduction, we will preprocess the data. We noticed that some of the columns had abnormal distributions and were skewed to one side. To normalize the data, we used the MinMaxScaler algorithm, which scales the data between 0 and 1. We also used PCA to reduce the dimensionality of the dataset.
```
#read the dataset

df = pd.read_csv("tripadvisor_review.csv")

#normalize the data
mms = MinMaxScaler()
mms.fit(df)
df = mms.transform(df)

#reduce the dimensionality of the dataset
n_components = df.shape[1]
pca = PCA(n_components=n_components, random_state=SEED)
pca.fit(df)
pca_df = pca.transform(df)
```
---

###  The Explained Variance Ratio
Here we are reducing the dimension of data in order to reduce the curse of dimensionality (an aspect where with an increase in data dimensions, dealing with it also becomes more complex). In this, we use the explained variance ratio to determine how many components are needed to explain the variance in the dataset.
The explained variance ratio is a measure of how much information (variance) can be attributed to each principal component. We found that the majority of the variance in the data can be encoded in 7 of the 10 dimensions, and over 72% of the variance can be encoded in 3 dimensions. Since we can work fine in 3D (but not for this blog), It suggests that we can expect to see some of the underlying structure in a 3D visualization although some information (less than 25%) will be hidden. Below is a snippet used to get the component representability together with final dimension reduction.
```
#determine the explained variance ratio
# get variance/data representability of each of the component
for i in range(n_components):
    first_n = pca.explained_variance_ratio_[0:i+1].sum()*100
    print(f'Percent variance explained by first {i+1} components: {round(first_n , 4)}%')

#reduce the dataset to 2 dimensions using PCA and visualize
pca = PCA(n_components=2, random_state=SEED)
pca.fit(df)
traindf = pca.transform(df)
plt.title("2D plot of the data representation")
sns.scatterplot(traindf[: , 0] , traindf[: , 1])
```
---

Data Clustering
After we have determined the number of components, the step that follows is clustering. Here, we actually don't know the number of clusters that our data should have. Due to this, we turn to the use of the elbow curve method and silhouette score to determine the optimal number of clusters.
Elbow Curve method.

We will fit our clustering algorithm using several different values of k, where k is the number of clusters, ranging from 2 to 15. For each value of k, we will evaluate the clustering results using the average sum of the squared distance score. We will plot the results against each k value and identify which number of clusters leads to the best results. Some of the steps and code snippet is as shown below.
Initialize an empty list to store the Sum_of_squared_distances for each value of k.
For each value of k, perform the k-means clustering algorithm on the data and calculate the Sum_of_squared_distances.
Append the Sum_of_squared_distances to the list.
Plot the Sum_of_squared_distances against the number of clusters, k.
Identify the elbow point in the plot, which is the optimal number of clusters.

```
Sum_of_squared_distances = []
for k in range(1,15):
    km = KMeans(n_clusters=k, init='k-means++')
    km = km.fit(traindf)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(range(1,15), Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
```
#### 2. Using Silhouette Score Method
The silhouette score is described as a measure of how close a sample is to members of its own cluster as compared to members of other clusters. The silhouette score ranges from -1 to 1. A score close to one indicates that a record is very close to other members of its cluster and far from members of other clusters. A score of 0 indicates that a record lies on the decision boundary between two clusters.
A negative score indicates that a sample is closer to members of a cluster other than its own. By taking the average silhouette score for all records when various numbers of clusters are used in our clustering algorithm, we can find the optimal number of clusters that promotes cohesion within individual clusters and good separability between the clusters.

```
max_k=15

sil_scores=[]
for i in range(2, max_k+1):
    clusterer = GaussianMixture(n_components=i, random_state=SEED, n_init=5)
    clusterer.fit(traindf)

    #Predict the cluster for each data point
    preds = clusterer.predict(traindf)

    #Find the cluster centers
    centers = clusterer.means_

    #Calculate the mean silhouette coefficient for the number of clusters chosen
    score = silhouette_score(traindf, preds)
    sil_scores.append(score)
    
sil_scores= pd.Series(sil_scores, index= range(2,max_k+1))
max_score= sil_scores.max()
n_clusters= sil_scores.idxmax()
print('Max Silhouette Score: {:.3f}'.format(max_score))
print('Number of clusters: {}\n'.format(max_k))

print('First 3 Silhouette Scores')
print(sil_scores[0:3])
```
---

### Comparing Results of Silhouette Score and Elbow Curve Methods
After using both the Silhouette score and Elbow curve methods to determine the optimal number of clusters for our data, we can now compare the results and visualize the clusters using the chosen number of clusters.
The silhouette score method shows that the maximum score is achieved when we use 2(this is the starting point for the analysis hence ignored) clusters which then went to 4. Therefore, we will fit the Gaussian Mixture Model (GMM) using 4 clusters. We will then color each cluster differently and visualize the data in 2 dimensions again using the first 2 components. This will allow us to see how the property we are interested in is distributed among our clusters.
On the other hand, the Elbow curve method shows that the optimal number of clusters could range from 2 to 4. Since the difference between the Silhouette score and Elbow curve methods is not significant, we will generalize between 2 and 5 as our preferred cluster numbers. From these results, we will use k as 2, 3, and 4 since they appear to be better.

---

### Comparing Results using Selected Clusters
We will now visualize our clusters and compare the results obtained from the two methods. We will use a scatter plot to show how the data points are distributed among the clusters. Each cluster will be represented by a different color, allowing us to easily identify the different clusters.
Using GMM model
```
f = plt.figure(figsize=(16 ,10))
plt.title("Comparison of various Cluster points using Gausian Mixture method" , fontsize =24)
f.add_subplot(2, 2, 1)
for i in [2,3,4 , 5]:
    model_i = GaussianMixture(n_components=i, random_state=2021, n_init=5)
    model_i.fit(traindf)
    label_i = model_i.predict(traindf)
    f.add_subplot(2, 2, i-1)
    sns.scatterplot(traindf[:, 0], traindf[:, 1] , c = label_i , label="n_cluster-"+str(i))
    plt.legend()
plt.show()
```
###  2. SpectralClustering algorithm
A graph of spectral Algorithm comparison for different k values3. AgglomerativeClustering Algorithm method
Agglomerative clustering is a type of hierarchical clustering algorithm that works by iteratively merging the closest pairs of clusters until a stopping criterion is reached. The algorithm starts with each data point in its own cluster and then merges the clusters based on linkage criteria such as complete linkage, single linkage, or average linkage.
To determine the optimal number of clusters for AgglomerativeClustering, we create a dendrogram that shows the hierarchical structure of the clusters. The dendrogram can help us to identify the appropriate number of clusters by visually inspecting the height of the vertical lines that represent the distance between the clusters. The code snippet is as shown below
```
def dendrogramPlot(model, **kwargs): 
    '''
    IT creates a dendogram plot using Agglomerative Clustering algorithm
    '''
    counts = np.zeros(model.children_.shape[0])

    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_,counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)  

ClusteringModel = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
# setting distance_threshold=0 ensures we compute the full tree.

ClusteringModel = ClusteringModel.fit(traindf)

plt.figure(figsize=(25,10))
plt.title('Clustering Dendrogram for 4 levels')
dendrogramPlot(ClusteringModel, p=4, truncate_mode='level')                    
plt.xlabel("Number of points in node")
plt.show()
```
---
Agglomerative clustering with Dendogram of level 5.The dendrogram plots generated using Hierarchical Agglomerative Clustering Algorithm suggest that the optimal number of clusters is 4, as both the level 4 and 5 dendrograms show a clear separation between the clusters. We can use this information to create a model with 4 clusters and visualize the results using a graph of the predicted class labels.
Aside above three models, the same process of training and visualizing was done in other models like DBSCAN, KMEANS, etc whose code can be found on the notebook link at the end of the blog.

---

### Model's Analysis and Results
In this analysis, we will be comparing the performance of different clustering algorithms on two different clusters (clusters 3 and 4). To evaluate the performance of each algorithm, we will be using two metrics i.e The Davies-Bouldin score and the silhouette score. The whole code structure for this comparison and visualization is available at the GitHub notebook.

#### Cluster 3 Results
The following are the results obtained for cluster 3:

````
          RESULTS METRICS COMPARISON FOR USING 3 CLUSTERS.
//=========================[]==============[]==================\\
|| Algorithm               || Davies Score || Silhouette Score ||
|]=========================[]==============[]==================[|
|| MiniBatchKmeans         || 0.883340     || 0.396502         ||
|| Kmeans++                || 0.888266     || 0.396317         ||
|| Gaussian_mixture        || 0.952638     || 0.388971         ||
|| SpectralClustering      || 0.909897     || 0.378901         ||
|| SpectralClusterer       || 0.879011     || 0.377372         ||
|| KMedoids_cosine         || 0.848656     || 0.366039         ||
|| AgglomerativeClustering || 0.938955     || 0.336498         ||
\\=========================[]==============[]==================//
````

MiniBatchKmeans was the best-performing algorithm when based on the silhouette score followed by Kmeans and then the Gaussian mixture. This cluster seems to be having different results of which model is the best.
For this purpose, MinibatchKmeans and Kmeans++ will be selected as the best model since they were first and second respectively when using silhouette score and second and third respectively when using the davies index

#### Cluster 4 Results
The following are the results obtained when using 4 clusters.
````
            RESULTS METRICS COMPARISON FOR USING 3 CLUSTERS.
//=========================[]==============[]==================\\
|| Algorithm               || Davies Score || Silhouette Score ||
|]=========================[]==============[]==================[|
|| Gaussian_mixture        || 0.776347     || 0.413012         ||
|| MiniBatchKmeans         || 0.813552     || 0.411235         ||
|| Kmeans++                || 0.834765     || 0.409727         ||
|| KMedoids_cosine         || 0.851633     || 0.372040         ||
|| SpectralClustering      || 0.959275     || 0.352476         ||
|| AgglomerativeClustering || 1.031960     || 0.313524         ||
|| SpectralClusterer       || 4.206309     || 0.226269         ||
\\=========================[]==============[]==================//

````
From the table above, we observe that the Gaussian mixture model had the lowest Davies score, while MiniBatchKmeans had the highest silhouette score. Overall, using various methods for clustering, we were able to obtain good results. The results showed the accurate metric in each algorithm, and based on the performance of each algorithm, we can select MiniBatchKmeans and Kmeans++ as the best models for cluster 3, and Gaussian mixture and MiniBatchKmeans as the best models for cluster 4.

---

### Conclusion
In conclusion, this analysis shows that unsupervised learning techniques such as data clustering are useful for analyzing and clustering high-dimensional data. We have explored different clustering algorithms and applied them to a real-world dataset of hotel reviews. By clustering the reviews, we can identify patterns and group similar reviews together. Our analysis shows that MiniBatchKmeans and Kmeans++ were the best-performing algorithms for Cluster 3 and 4 based on Silhouette Score and Davies Bouldin Index.
We also observed that the quality of the reviews in Cluster 4 was generally higher than those in Cluster 3, with a higher Silhouette Score and lower Davies Bouldin Index. The labels assigned to each review provided useful insights into the overall quality of the hotels, with a rating of 4 indicating excellent reviews and 0 indicating terrible reviews. Unsupervised learning techniques such as clustering can be a powerful tool for analyzing high-dimensional data and extracting meaningful insights.

---

The code used for the blog can be found in the following repository.
GitHub …
You can't perform that action at this time. You signed in with another tab or window. You signed out in another tab or…github.com
