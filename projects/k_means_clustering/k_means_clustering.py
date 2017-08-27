import numpy as np
from bokeh.plotting import figure, show, output_file
from sklearn import datasets
from sklearn.cluster import KMeans

np.random.seed(3)

# get the dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# fit K-means clustering
est = KMeans(n_clusters=3).fit(X)


# create scatterplot with each cluster in a specific colour

colormap = {0: 'red', 1: 'green', 2: 'blue'}
color_per_cluster = [colormap[y] for y in est.labels_.astype(np.float)]
p = figure(title="Iris Morphology")
p.xaxis.axis_label = 'Petal Length'
p.yaxis.axis_label = 'Petal Width'
p.circle(X[:, 2], X[:, 3],
         color=color_per_cluster, fill_alpha=0.2, size=10)

output_file("iris.html", title="iris.py example")
show(p)
