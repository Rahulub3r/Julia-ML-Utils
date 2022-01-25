# Julia-ML-Utils

These are my own minimum working examples of some utiltiies for machine learning that are not available in `MLJ` or `Clustering`.

The following are contained:

## 1 - generalUtils.jl

A function for Cohen's kappa

## 2 - clusteringUtils.jl

A function to calculate dynamic time warping (DTW) distance using a Sakoe-chiba band and a symmetricP0 pattern recursion. Use as follows:

```
# 3 below is the window
dtwCTM(series1, series2, 3; dist_fn = _sq_dist)
```

A function to plot dendrogram using CairoMakie.jl. Use as follows:

```
using Clustering
using CairoMakie

num_clusters = 6 # number of features to cut at
num_features = 10 #number of features in the dataframe

fig = Figure(resolution = (600, 500))
ax1 = Axis(fig[1,1], limits=(-15, nothing, nothing, nothing))

node_locations = drawDendrogram(some_hclust, ax1; feature_names = nothing)
vlines!(ax1, node_locations[num_features-num_clusters][1] + 0.5)

fig
```

## 3 - decisionTreeUtils.jl

Contains a function to plot decision tree using CairoMakie.jl. Usage is shown in the code chunk below.

Customizations can be done using arguments:

- `nodewth` - width of a node. can be increased or decreased based on plot resolution. (default = 10)
- `nodeht` - height of a node. can be increased or decreased based on plot resolution (default = 4)
- `nodeboxcolor` - can specify a color for the node (default = Makie.wong_colors()[1])
- `leafboxcolorpalette` - can specify a color palette for all the leaves. (default = Makie.wong_colors())
- `boxbordercolor` - self-explanatory (default = :black)
- `nodetextsize` - self-explanatory (default=10)
- `nodetextcolor` - self-explanatory (default=:black)
- `leaftextsize` - self-explanatory (default=10)
- `leaftextcolor` - self-explanatory (default=:black)
- `leafwth` - width of leaves (default = 7)
- `leafht` - height of leaves (default = 4)
- `linetextcolor` - Color of the text on lines (default = :black)
- `linetextsize` - Size of the text on lines (default = 10)
- `feature_names` - vector of feature names. (default = nothing)

```
using MLJ
using DecisionTree
using CairoMakie

X, y = make_blobs()

dtc = @load DecisionTreeClassifier pkg=DecisionTree verbosity=0
dtc_model = dtc(min_purity_increase=0.005, min_samples_leaf=1,         min_samples_split=2, max_depth=6)
dtc_mach = machine(dtc_model, X, y)
MLJ.fit!(dtc_mach)
x = fitted_params(dtc_mach)
print_tree(x.tree)

f = Figure(;resolution=(1600, 600))
ax1 = Axis(f[1,1])
drawTree(x.tree, x.encoding, ax1; feature_names=["X1", "X2"], 
        nodetextsize=15, nodetextcolor=:black,
        linetextsize=13, leaftextsize=13, leafwth=4)
hidespines!(ax1)
hidedecorations!(ax1)
f

```