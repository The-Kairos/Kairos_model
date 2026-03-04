# Retrieval Comparison: How to Make Pasta - Without a Machine.mp4

## Config

- **k**: 5

- **top_c**: 2

- **alpha**: 0.3

- **cluster_k**: 8


## Cluster counts

- KMeans: 8 clusters

- HDBSCAN: 2 clusters

## Summary

| Method | Avg Time (s) | Avg Jaccard vs Flat | Notes |

|---|---:|---:|---:|

| flat | 0.000284 | 0.000 |  |

| kmeans | 0.010352 | 0.756 |  |

| hdbscan | 0.008638 | 0.633 |  |


## Per-query details

### Query: the scene where the person washes their hands

- **flat**: time=0.000367s, top_indices=[5, 4, 8, 7, 48], scores=[0.718, 0.705, 0.652, 0.642, 0.636]

- **kmeans**: time=0.010596s, top_indices=[5, 4, 8, 7, 44], scores=[1.018, 1.005, 0.952, 0.942, 0.917]

- **hdbscan**: time=0.009400s, top_indices=[8, 7, 39, 32, 25], scores=[0.952, 0.941, 0.933, 0.929, 0.925]



### Query: the scene where they are rolling dough

- **flat**: time=0.000232s, top_indices=[28, 22, 35, 27, 32], scores=[0.732, 0.722, 0.721, 0.716, 0.715]

- **kmeans**: time=0.008914s, top_indices=[28, 22, 35, 27, 32], scores=[1.032, 1.022, 1.021, 1.016, 1.015]

- **hdbscan**: time=0.008456s, top_indices=[28, 22, 35, 27, 32], scores=[1.032, 1.022, 1.021, 1.016, 1.015]



### Query: the scene where flour is being measured

- **flat**: time=0.000279s, top_indices=[6, 8, 7, 13, 10], scores=[0.744, 0.722, 0.714, 0.708, 0.706]

- **kmeans**: time=0.011066s, top_indices=[6, 8, 7, 13, 10], scores=[1.043, 1.021, 1.013, 1.008, 1.006]

- **hdbscan**: time=0.008364s, top_indices=[6, 8, 7, 13, 10], scores=[1.044, 1.022, 1.014, 1.008, 1.006]



### Query: when they serve pasta

- **flat**: time=0.000291s, top_indices=[3, 43, 2, 50, 49], scores=[0.636, 0.632, 0.632, 0.632, 0.631]

- **kmeans**: time=0.011503s, top_indices=[3, 43, 2, 50, 49], scores=[0.936, 0.932, 0.932, 0.932, 0.931]

- **hdbscan**: time=0.008619s, top_indices=[2, 49, 45, 51, 46], scores=[0.932, 0.931, 0.915, 0.906, 0.901]



### Query: a cooking demonstration

- **flat**: time=0.000252s, top_indices=[45, 0, 58, 36, 2], scores=[0.629, 0.62, 0.619, 0.618, 0.617]

- **kmeans**: time=0.009679s, top_indices=[45, 46, 42, 44, 54], scores=[0.929, 0.915, 0.915, 0.912, 0.908]

- **hdbscan**: time=0.008350s, top_indices=[45, 0, 2, 36, 46], scores=[0.927, 0.918, 0.917, 0.917, 0.914]


