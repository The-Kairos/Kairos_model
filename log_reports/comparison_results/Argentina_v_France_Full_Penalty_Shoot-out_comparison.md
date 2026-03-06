# Retrieval Comparison: Argentina v France Full Penalty Shoot-out.mp4
## Config
- **k**: 5
- **top_c**: 2
- **alpha**: 0.3

## Cluster counts
- KMeans: 3 clusters
- HDBSCAN: 2 clusters
## Summary
| Method | Avg Time (s) | Avg Jaccard vs Flat | Notes |
|---|---:|---:|---:|
| flat | 0.000832 | 0.000 |  |
| kmeans | 0.012794 | 0.752 |  |
| hdbscan | 0.011626 | 0.539 |  |

## Per-query details
### Query: Give me the clip where Messi scores
- **flat**: time=0.003152s, top_indices=[68, 77, 67, 55, 10], scores=[0.705, 0.704, 0.698, 0.693, 0.687]
- **kmeans**: time=0.016557s, top_indices=[68, 77, 67, 71, 78], scores=[1.005, 1.002, 0.998, 0.984, 0.982]
- **hdbscan**: time=0.011578s, top_indices=[68, 67, 55, 71, 10], scores=[1.005, 0.998, 0.989, 0.984, 0.983]

#### Scene Details
| Scene Index | Timestamp | Description |
|---|---|---|
| 10 | 00:00:59.200 | This scene captures a moment during the penalty shootout in the high-stakes soccer match between Argentina and France. The video shows a soccer player |
| 55 | 00:05:40.320 | This scene captures a critical moment during the penalty shootout in the intense soccer match between Argentina and France. The video shows a soccer p |
| 67 | 00:06:49.920 | This scene captures a climactic moment during the World Cup final penalty shootout between Argentina and France. The video shows players and fans inte |
| 68 | 00:07:04.320 | This scene captures a celebratory moment during the World Cup final penalty shootout between Argentina and France. The video shows Lionel Messi and hi |
| 71 | 00:07:17.440 | Summary: This scene captures a celebratory moment during the World Cup final penalty shootout between Argentina and France. The video shows a soccer p |

### Query: Where the Argentinian goalkeeper Martinez blocks the goal
- **flat**: time=0.000255s, top_indices=[26, 18, 17, 76, 74], scores=[0.73, 0.722, 0.704, 0.691, 0.691]
- **kmeans**: time=0.012434s, top_indices=[26, 18, 76, 74, 78], scores=[1.019, 1.011, 0.991, 0.991, 0.989]
- **hdbscan**: time=0.011659s, top_indices=[26, 45, 7, 44, 36], scores=[1.03, 0.987, 0.984, 0.982, 0.979]

#### Scene Details
| Scene Index | Timestamp | Description |
|---|---|---|
| 7 | 00:00:36.960 | This scene captures a critical moment in the penalty shootout during the high-stakes soccer match between Argentina and France. The video shows a goal |
| 17 | 00:01:55.040 | This scene captures a pivotal moment during the penalty shootout in the high-stakes soccer match between Argentina and France. The video shows players |
| 18 | 00:02:05.280 | This scene captures a moment during the penalty shootout in the intense soccer match between Argentina and France. The video shows a stationary goalke |
| 26 | 00:03:05.600 | This scene captures a moment during the penalty shootout in the intense soccer match between Argentina and France. The video shows a soccer stadium fi |
| 36 | 00:03:51.200 | This scene captures a moment during the penalty shootout in the intense soccer match between Argentina and France. The video shows players on the fiel |
| 44 | 00:04:31.840 | This scene captures a moment during the penalty shootout in the intense soccer match between Argentina and France. The video shows a soccer goalie nea |
| 45 | 00:04:35.840 | This scene captures a moment during the penalty shootout in the intense soccer match between Argentina and France. The video shows a soccer goalie nea |

### Query: Commentators speaking about Kylian Mbappe failing at the European championships against Switzerland
- **flat**: time=0.000171s, top_indices=[2, 1, 31, 4, 25], scores=[0.764, 0.694, 0.675, 0.668, 0.662]
- **kmeans**: time=0.010939s, top_indices=[2, 1, 31, 4, 25], scores=[1.064, 0.994, 0.975, 0.968, 0.962]
- **hdbscan**: time=0.011308s, top_indices=[2, 1, 31, 25, 16], scores=[1.064, 0.994, 0.975, 0.962, 0.956]

#### Scene Details
| Scene Index | Timestamp | Description |
|---|---|---|
| 1 | 00:00:04.800 | This scene captures the continuation of the penalty shootout during a high-stakes soccer match between Argentina and France. The video shows close-up |
| 2 | 00:00:11.040 | This scene captures a critical moment in the penalty shootout during a high-stakes soccer match between Argentina and France. The video shows players |
| 4 | 00:00:24.800 | This scene captures the continuation of the penalty shootout in the high-stakes soccer match between Argentina and France. The video shows the atmosph |
| 16 | 00:01:48.000 | This scene captures a moment during the penalty shootout in the intense soccer match between Argentina and France. The video shows players positioned |
| 25 | 00:02:57.920 | This scene captures a moment during the penalty shootout in the intense soccer match between Argentina and France. The video shows players moving on t |
| 31 | 00:03:32.640 | This scene captures a moment during the penalty shootout in the intense soccer match between Argentina and France. The video shows a soccer player on |

### Query: The Argentinian team celebrating and hugging on their victory
- **flat**: time=0.000273s, top_indices=[57, 62, 61, 58, 68], scores=[0.75, 0.74, 0.738, 0.737, 0.737]
- **kmeans**: time=0.012133s, top_indices=[57, 62, 61, 58, 68], scores=[1.05, 1.04, 1.038, 1.037, 1.037]
- **hdbscan**: time=0.011770s, top_indices=[57, 62, 61, 58, 68], scores=[1.05, 1.04, 1.038, 1.037, 1.037]

#### Scene Details
| Scene Index | Timestamp | Description |
|---|---|---|
| 57 | 00:05:53.600 | This scene captures the climactic moment of the penalty shootout in the intense soccer match between Argentina and France. The video shows the Argenti |
| 58 | 00:05:57.280 | This scene captures a celebratory moment following the conclusion of the penalty shootout in the intense soccer match between Argentina and France. Th |
| 61 | 00:06:13.120 | This scene captures the celebratory atmosphere following Argentina's victory in the penalty shootout against France during the World Cup final. The vi |
| 62 | 00:06:25.120 | This scene captures the climactic moment of Argentina's victory in the World Cup final penalty shootout against France. The video shows jubilant playe |
| 68 | 00:07:04.320 | This scene captures a celebratory moment during the World Cup final penalty shootout between Argentina and France. The video shows Lionel Messi and hi |

### Query: Retrieve the scenes where a crowd is shown
- **flat**: time=0.000307s, top_indices=[0, 12, 59, 72, 6], scores=[0.719, 0.712, 0.706, 0.691, 0.689]
- **kmeans**: time=0.011906s, top_indices=[0, 12, 59, 6, 11], scores=[1.013, 1.006, 1.0, 0.989, 0.987]
- **hdbscan**: time=0.011815s, top_indices=[0, 59, 67, 29, 63], scores=[1.014, 1.001, 0.983, 0.982, 0.981]

#### Scene Details
| Scene Index | Timestamp | Description |
|---|---|---|
| 0 | 00:00:00.000 | This scene captures the atmosphere of a soccer stadium during a high-stakes penalty shootout. The video shows a view of the stadium with fans in the s |
| 6 | 00:00:33.280 | This scene captures a moment of celebration and tension during the penalty shootout in the high-stakes soccer match between Argentina and France. The |
| 11 | 00:01:17.280 | This scene captures a moment during the penalty shootout in the intense soccer match between Argentina and France. The video shows various individuals |
| 12 | 00:01:20.960 | This scene captures a moment during the penalty shootout in the soccer match between Argentina and France. The video shows a crowd watching the game, |
| 29 | 00:03:16.640 | This scene captures a moment during the penalty shootout in the intense soccer match between Argentina and France. The video shows a soccer player hol |
| 59 | 00:06:06.400 | This scene captures the atmosphere of a soccer stadium during the World Cup final penalty shootout between Argentina and France. The video shows the s |
| 63 | 00:06:34.720 | This scene captures a celebratory moment as Argentina secures victory in the World Cup final penalty shootout against France. The video shows players |
| 67 | 00:06:49.920 | This scene captures a climactic moment during the World Cup final penalty shootout between Argentina and France. The video shows players and fans inte |
| 72 | 00:07:23.520 | Summary:   This scene captures a moment of confusion and inconsistency unrelated to the World Cup final penalty shootout between Argentina and France. |

