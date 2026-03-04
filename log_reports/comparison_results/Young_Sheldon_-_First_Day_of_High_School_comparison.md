# Retrieval Comparison: Young Sheldon - First Day of High School.mp4

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

| flat | 0.000255 | 0.000 |  |

| kmeans | 0.006847 | 0.686 |  |

| hdbscan | 0.005633 | 0.266 |  |


## Per-query details

### Query: Give me the clip of the boy entering his class

- **flat**: time=0.000183s, top_indices=[26, 22, 34, 25, 21], scores=[0.726, 0.714, 0.71, 0.704, 0.697]

- **kmeans**: time=0.006355s, top_indices=[34, 25, 38, 26, 22], scores=[1.009, 1.004, 0.993, 0.726, 0.714]

- **hdbscan**: time=0.005429s, top_indices=[26, 22, 21, 14, 15], scores=[1.026, 1.014, 0.997, 0.988, 0.983]



### Query: Give me the scene of the mom worried

- **flat**: time=0.000251s, top_indices=[9, 16, 18, 26, 22], scores=[0.732, 0.705, 0.685, 0.683, 0.683]

- **kmeans**: time=0.007294s, top_indices=[9, 16, 18, 26, 22], scores=[1.032, 1.005, 0.985, 0.983, 0.983]

- **hdbscan**: time=0.005512s, top_indices=[16, 18, 19, 17, 14], scores=[1.005, 0.985, 0.979, 0.976, 0.976]



### Query: Show me the scenes that have music

- **flat**: time=0.000184s, top_indices=[33, 0, 2, 19, 1], scores=[0.695, 0.688, 0.682, 0.663, 0.658]

- **kmeans**: time=0.006977s, top_indices=[33, 0, 2, 1, 10], scores=[0.995, 0.988, 0.982, 0.941, 0.934]

- **hdbscan**: time=0.006044s, top_indices=[19, 17, 18, 32, 21], scores=[0.963, 0.939, 0.929, 0.928, 0.928]



### Query: Show me clips of the school

- **flat**: time=0.000292s, top_indices=[38, 37, 22, 34, 20], scores=[0.649, 0.648, 0.646, 0.643, 0.639]

- **kmeans**: time=0.006047s, top_indices=[38, 34, 25, 37, 22], scores=[0.949, 0.943, 0.933, 0.648, 0.646]

- **hdbscan**: time=0.005599s, top_indices=[22, 26, 21, 32, 17], scores=[0.946, 0.937, 0.936, 0.935, 0.935]



### Query: A clip where students are in a classroom

- **flat**: time=0.000366s, top_indices=[26, 20, 25, 22, 15], scores=[0.697, 0.688, 0.687, 0.684, 0.683]

- **kmeans**: time=0.007561s, top_indices=[26, 25, 22, 27, 21], scores=[0.994, 0.987, 0.981, 0.977, 0.974]

- **hdbscan**: time=0.005580s, top_indices=[26, 22, 15, 21, 31], scores=[0.997, 0.984, 0.981, 0.977, 0.976]


