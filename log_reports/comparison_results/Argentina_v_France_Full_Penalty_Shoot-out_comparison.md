# Retrieval Comparison: Argentina v France Full Penalty Shoot-out.mp4

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

| flat | 0.000648 | 0.000 |  |

| kmeans | 0.012614 | 0.586 |  |

| hdbscan | 0.013499 | 0.539 |  |


## Per-query details

### Query: Give me the clip where Messi scores

- **flat**: time=0.002373s, top_indices=[68, 77, 67, 55, 10], scores=[0.705, 0.704, 0.698, 0.693, 0.687]

- **kmeans**: time=0.014678s, top_indices=[68, 67, 71, 78, 44], scores=[1.005, 0.998, 0.984, 0.984, 0.98]

- **hdbscan**: time=0.010306s, top_indices=[68, 67, 55, 71, 10], scores=[1.005, 0.998, 0.989, 0.984, 0.983]



### Query: Where the Argentinian goalkeeper Martinez blocks the goal

- **flat**: time=0.000182s, top_indices=[26, 18, 17, 76, 74], scores=[0.73, 0.722, 0.704, 0.691, 0.691]

- **kmeans**: time=0.012223s, top_indices=[76, 74, 45, 44, 77], scores=[0.991, 0.991, 0.983, 0.978, 0.97]

- **hdbscan**: time=0.024766s, top_indices=[26, 45, 7, 44, 36], scores=[1.03, 0.987, 0.984, 0.982, 0.979]



### Query: Commentators speaking about Kylian Mbappe failing at the European championships against Switzerland

- **flat**: time=0.000248s, top_indices=[2, 1, 31, 4, 25], scores=[0.764, 0.694, 0.675, 0.668, 0.662]

- **kmeans**: time=0.013193s, top_indices=[2, 1, 31, 4, 25], scores=[1.064, 0.993, 0.975, 0.967, 0.961]

- **hdbscan**: time=0.010434s, top_indices=[2, 1, 31, 25, 16], scores=[1.064, 0.994, 0.975, 0.962, 0.956]



### Query: The Argentinian team celebrating and hugging on their victory

- **flat**: time=0.000206s, top_indices=[57, 62, 61, 58, 68], scores=[0.75, 0.74, 0.738, 0.737, 0.737]

- **kmeans**: time=0.012127s, top_indices=[57, 62, 61, 58, 68], scores=[1.05, 1.04, 1.038, 1.037, 1.037]

- **hdbscan**: time=0.011271s, top_indices=[57, 62, 61, 58, 68], scores=[1.05, 1.04, 1.038, 1.037, 1.037]



### Query: Retrieve the scenes where a crowd is shown

- **flat**: time=0.000230s, top_indices=[0, 12, 59, 72, 6], scores=[0.719, 0.712, 0.706, 0.691, 0.689]

- **kmeans**: time=0.010851s, top_indices=[6, 11, 72, 73, 0], scores=[0.989, 0.987, 0.987, 0.963, 0.719]

- **hdbscan**: time=0.010718s, top_indices=[0, 59, 67, 29, 63], scores=[1.014, 1.001, 0.983, 0.982, 0.981]


