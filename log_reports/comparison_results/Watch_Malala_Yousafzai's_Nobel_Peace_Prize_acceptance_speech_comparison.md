# Retrieval Comparison: Watch Malala Yousafzai's Nobel Peace Prize acceptance speech.mp4
## Config
- **k**: 5
- **top_c**: 2
- **alpha**: 0.3

## Cluster counts
- KMeans: 4 clusters
- HDBSCAN: 0 clusters
## Summary
| Method | Avg Time (s) | Avg Jaccard vs Flat | Notes |
|---|---:|---:|---:|
| flat | 0.000208 | 0.000 |  |
| kmeans | 0.005439 | 0.933 |  |
| hdbscan | 0.003948 | 1.000 |  |

## Per-query details
### Query: Give me the clip of the woman with a colorful hijab sitting next to a man
- **flat**: time=0.000180s, top_indices=[7, 9, 21, 19, 11], scores=[0.688, 0.675, 0.672, 0.667, 0.663]
- **kmeans**: time=0.004117s, top_indices=[7, 9, 21, 19, 14], scores=[0.988, 0.975, 0.972, 0.964, 0.959]
- **hdbscan**: time=0.003843s, top_indices=[7, 9, 21, 19, 11], scores=[0.688, 0.675, 0.672, 0.667, 0.663]

#### Scene Details
| Scene Index | Timestamp | Description |
|---|---|---|
| 7 | 00:00:59.526 | <short scene paragraph>   The scene shows Malala Yousafzai, wearing a pink hijab, speaking at a podium during her Nobel Peace Prize acceptance speech. |
| 9 | 00:01:11.271 | <short scene paragraph>   The scene shows Malala Yousafzai continuing her Nobel Peace Prize acceptance speech. She is seen wearing a red scarf and spe |
| 11 | 00:01:33.694 | <short scene paragraph>   The scene shows Malala Yousafzai, wearing a red scarf, speaking at a podium during her Nobel Peace Prize acceptance speech. |
| 14 | 00:01:42.502 | <short scene paragraph>   The scene shows Malala Yousafzai speaking at a podium during her Nobel Peace Prize acceptance speech. She is wearing a red s |
| 19 | 00:03:30.210 | <short scene paragraph>   The scene shows two men in a room with flowers and a sign, sitting and talking to each other. The audio includes applause an |
| 21 | 00:03:36.750 | The scene shows Malala Yousafzai standing at a podium in a formal event setting, wearing pink, addressing an audience. She speaks about how people per |

### Query: Give me the clip of Kailash Satyarthi wearing glasses and white clothes clapping for Malala
- **flat**: time=0.000282s, top_indices=[13, 18, 12, 17, 2], scores=[0.744, 0.744, 0.743, 0.742, 0.741]
- **kmeans**: time=0.009904s, top_indices=[13, 18, 12, 17, 2], scores=[1.044, 1.044, 1.043, 1.042, 1.041]
- **hdbscan**: time=0.004270s, top_indices=[13, 18, 12, 17, 2], scores=[0.744, 0.744, 0.743, 0.742, 0.741]

#### Scene Details
| Scene Index | Timestamp | Description |
|---|---|---|
| 2 | 00:00:21.755 | Scene Report:  Characters:   Name: Malala Yousafzai   Role: Speaker, Nobel Peace Prize recipient    Key Dialogue:   Speaker: Malala Yousafzai   Cause: |
| 12 | 00:01:36.363 | <short scene paragraph>   The scene shows an audience clapping and smiling during an event. The visuals suggest a formal setting, though there are inc |
| 13 | 00:01:38.765 | Scene Report:  The scene shows an audience clapping in a formal event setting, with applause and a brief "Thank you" heard in the audio. The visuals a |
| 17 | 00:03:24.337 | <short scene paragraph>   The scene shows an audience clapping during a formal event, with applause and a brief "Thank you" heard in the audio. The vi |
| 18 | 00:03:27.674 | The scene shows an audience clapping during a formal event, with applause and a brief "Thank you" heard in the audio. The visuals are inconsistent, wi |

### Query: Give me the clip of a room full of people clapping
- **flat**: time=0.000196s, top_indices=[13, 12, 18, 17, 20], scores=[0.691, 0.689, 0.683, 0.68, 0.672]
- **kmeans**: time=0.004074s, top_indices=[13, 12, 18, 17, 20], scores=[0.991, 0.989, 0.983, 0.98, 0.972]
- **hdbscan**: time=0.003711s, top_indices=[13, 12, 18, 17, 20], scores=[0.691, 0.689, 0.683, 0.68, 0.672]

#### Scene Details
| Scene Index | Timestamp | Description |
|---|---|---|
| 12 | 00:01:36.363 | <short scene paragraph>   The scene shows an audience clapping and smiling during an event. The visuals suggest a formal setting, though there are inc |
| 13 | 00:01:38.765 | Scene Report:  The scene shows an audience clapping in a formal event setting, with applause and a brief "Thank you" heard in the audio. The visuals a |
| 17 | 00:03:24.337 | <short scene paragraph>   The scene shows an audience clapping during a formal event, with applause and a brief "Thank you" heard in the audio. The vi |
| 18 | 00:03:27.674 | The scene shows an audience clapping during a formal event, with applause and a brief "Thank you" heard in the audio. The visuals are inconsistent, wi |
| 20 | 00:03:33.413 | <short scene paragraph>   The scene shows a formal event with an audience clapping and a brief "Thank you" heard in the audio. The visuals inconsisten |

### Query: Where Malala says what her brothers call her
- **flat**: time=0.000216s, top_indices=[21, 25, 23, 4, 22], scores=[0.725, 0.722, 0.711, 0.698, 0.697]
- **kmeans**: time=0.004697s, top_indices=[25, 21, 23, 22, 4], scores=[1.022, 1.016, 1.011, 0.997, 0.988]
- **hdbscan**: time=0.004124s, top_indices=[21, 25, 23, 4, 22], scores=[0.725, 0.722, 0.711, 0.698, 0.697]

#### Scene Details
| Scene Index | Timestamp | Description |
|---|---|---|
| 4 | 00:00:28.562 | <short scene paragraph>   Malala Yousafzai continues her Nobel Peace Prize acceptance speech, blending humor and humility as she reflects on her perso |
| 21 | 00:03:36.750 | The scene shows Malala Yousafzai standing at a podium in a formal event setting, wearing pink, addressing an audience. She speaks about how people per |

### Query: Give me the clip where Malala fixes her pink hijab
- **flat**: time=0.000169s, top_indices=[7, 21, 25, 22, 26], scores=[0.74, 0.736, 0.728, 0.727, 0.709]
- **kmeans**: time=0.004402s, top_indices=[7, 21, 25, 22, 26], scores=[1.04, 1.035, 1.028, 1.027, 1.009]
- **hdbscan**: time=0.003795s, top_indices=[7, 21, 25, 22, 26], scores=[0.74, 0.736, 0.728, 0.727, 0.709]

#### Scene Details
| Scene Index | Timestamp | Description |
|---|---|---|
| 7 | 00:00:59.526 | <short scene paragraph>   The scene shows Malala Yousafzai, wearing a pink hijab, speaking at a podium during her Nobel Peace Prize acceptance speech. |
| 21 | 00:03:36.750 | The scene shows Malala Yousafzai standing at a podium in a formal event setting, wearing pink, addressing an audience. She speaks about how people per |

