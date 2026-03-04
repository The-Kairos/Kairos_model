# Retrieval Comparison: Watch Malala Yousafzai's Nobel Peace Prize acceptance speech.mp4

## Config

- **k**: 5

- **top_c**: 2

- **alpha**: 0.3

- **cluster_k**: 8


## Cluster counts

- KMeans: 8 clusters

- HDBSCAN: 0 clusters

## Summary

| Method | Avg Time (s) | Avg Jaccard vs Flat | Notes |

|---|---:|---:|---:|

| flat | 0.000218 | 0.000 |  |

| kmeans | 0.005640 | 0.602 |  |

| hdbscan | 0.003821 | 1.000 |  |


## Per-query details

### Query: Give me the clip of the woman with a colorful hijab sitting next to a man

- **flat**: time=0.000272s, top_indices=[7, 9, 21, 19, 11], scores=[0.688, 0.675, 0.672, 0.667, 0.663]

- **kmeans**: time=0.005877s, top_indices=[7, 9, 21, 14, 4], scores=[0.988, 0.975, 0.969, 0.959, 0.945]

- **hdbscan**: time=0.004093s, top_indices=[7, 9, 21, 19, 11], scores=[0.688, 0.675, 0.672, 0.667, 0.663]



### Query: Give me the clip of Kailash Satyarthi wearing glasses and white clothes clapping for Malala

- **flat**: time=0.000224s, top_indices=[13, 18, 12, 17, 2], scores=[0.744, 0.744, 0.743, 0.742, 0.741]

- **kmeans**: time=0.005531s, top_indices=[13, 18, 12, 17, 20], scores=[1.044, 1.044, 1.043, 1.042, 1.037]

- **hdbscan**: time=0.003420s, top_indices=[13, 18, 12, 17, 2], scores=[0.744, 0.744, 0.743, 0.742, 0.741]



### Query: Give me the clip of a room full of people clapping

- **flat**: time=0.000146s, top_indices=[13, 12, 18, 17, 20], scores=[0.691, 0.689, 0.683, 0.68, 0.672]

- **kmeans**: time=0.004968s, top_indices=[13, 12, 18, 17, 20], scores=[0.991, 0.989, 0.983, 0.98, 0.972]

- **hdbscan**: time=0.003851s, top_indices=[13, 12, 18, 17, 20], scores=[0.691, 0.689, 0.683, 0.68, 0.672]



### Query: Where Malala says what her brothers call her

- **flat**: time=0.000227s, top_indices=[21, 25, 23, 4, 22], scores=[0.725, 0.722, 0.711, 0.698, 0.697]

- **kmeans**: time=0.005764s, top_indices=[25, 21, 23, 4, 16], scores=[1.022, 1.013, 1.011, 0.985, 0.924]

- **hdbscan**: time=0.003585s, top_indices=[21, 25, 23, 4, 22], scores=[0.725, 0.722, 0.711, 0.698, 0.697]



### Query: Give me the clip where Malala fixes her pink hijab

- **flat**: time=0.000223s, top_indices=[7, 21, 25, 22, 26], scores=[0.74, 0.736, 0.728, 0.727, 0.709]

- **kmeans**: time=0.006059s, top_indices=[21, 25, 16, 23, 4], scores=[1.035, 1.028, 0.998, 0.995, 0.992]

- **hdbscan**: time=0.004156s, top_indices=[7, 21, 25, 22, 26], scores=[0.74, 0.736, 0.728, 0.727, 0.709]


