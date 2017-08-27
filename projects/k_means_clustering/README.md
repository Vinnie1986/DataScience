# K-means clustering

## BEFORE USAGE

run the following:
`pip install -r requirements.txt`


## WHAT?

K-means is a unsupervised learning method that groups your data in a pre-defined number of clusters.
We have 2 inputs:

1) The number of clusters we would like to have.
2) The dataset with features for each datapoint. 


We start with randomly choosing some datapoints equal to the number of clusters we have.
we assign each randomly selected datapoint to a cluster, we call this the centroid.

The clustering is based on 2 steps:

1) We assign each datapoint to a cluster. the selection is based on the minimization of the euclidean distance.
2) Update the centroid. we set the centroid to the average value of each cluster.

we iterate over these two steps until no more datapoints have changed cluster.

### tags

k-means, clustering, bokeh