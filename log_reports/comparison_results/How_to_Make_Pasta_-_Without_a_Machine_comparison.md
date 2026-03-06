# Retrieval Comparison: How to Make Pasta - Without a Machine.mp4
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
| flat | 0.000278 | 0.000 |  |
| kmeans | 0.011297 | 0.886 |  |
| hdbscan | 0.009327 | 0.633 |  |

## Per-query details
### Query: the scene where the person washes their hands
- **flat**: time=0.000256s, top_indices=[5, 4, 8, 7, 48], scores=[0.718, 0.705, 0.652, 0.642, 0.636]
- **kmeans**: time=0.010496s, top_indices=[5, 4, 8, 7, 48], scores=[1.018, 1.003, 0.952, 0.942, 0.935]
- **hdbscan**: time=0.008856s, top_indices=[8, 7, 39, 32, 25], scores=[0.952, 0.941, 0.933, 0.929, 0.925]

#### Scene Details
| Scene Index | Timestamp | Description |
|---|---|---|
| 4 | 00:00:18.167 | The scene begins with a man washing his hands in a sink using soap. The audio introduces the tutorial with an encouraging tone, stating that making pa |
| 5 | 00:00:22.833 | The scene shows the continuation of the pasta-making process, focusing on hygiene and preparation. The visuals depict a wooden table as the primary wo |
| 7 | 00:00:31.833 | <short scene paragraph>   The scene begins with a focus on the pasta-making process. The visuals show flour being used on a tabletop, with a person’s |
| 8 | 00:00:36.000 | <short scene paragraph>   The scene focuses on the early stages of pasta preparation, showing a person working with flour on a wooden table. The visua |
| 25 | 00:02:17.000 | Scene Report:  The scene shows a person working with dough as part of the pasta-making tutorial. Visuals depict hands holding dough, though captions a |
| 32 | 00:03:00.500 | The scene shows a person manually rolling out dough on a wooden table, with the goal of making it thin enough to see the color of their fingers throug |
| 39 | 00:03:48.167 | Scene Report:  The scene shows a person manually cutting pasta on a wooden cutting board with a knife, continuing the hands-on process of shaping the |
| 48 | 00:04:22.500 | The scene shows a person adding sauce to a bowl of pasta, followed by the addition of arugula. The visuals depict hands interacting with the pasta dis |

### Query: the scene where they are rolling dough
- **flat**: time=0.000305s, top_indices=[28, 22, 35, 27, 32], scores=[0.732, 0.722, 0.721, 0.716, 0.715]
- **kmeans**: time=0.008641s, top_indices=[28, 22, 35, 27, 32], scores=[1.032, 1.022, 1.021, 1.016, 1.015]
- **hdbscan**: time=0.009232s, top_indices=[28, 22, 35, 27, 32], scores=[1.032, 1.022, 1.021, 1.016, 1.015]

#### Scene Details
| Scene Index | Timestamp | Description |
|---|---|---|
| 22 | 00:02:01.000 | The scene shows a person rolling out dough on a wooden board placed on a wooden table. The visuals focus on the tactile process of flattening the doug |
| 27 | 00:02:24.167 | The scene shows a person working with dough on a wooden table. The visuals depict actions such as rolling and kneading dough, though captions are inco |
| 28 | 00:02:29.333 | The scene shows a person kneading and rolling dough on a wooden table dusted with flour. The audio provides clear instructions, stating, "We're going |
| 32 | 00:03:00.500 | The scene shows a person manually rolling out dough on a wooden table, with the goal of making it thin enough to see the color of their fingers throug |
| 35 | 00:03:13.667 | Scene Report:  The scene shows a person rolling out dough on a wooden table using a rolling pin, continuing the hands-on process of preparing pasta do |

### Query: the scene where flour is being measured
- **flat**: time=0.000227s, top_indices=[6, 8, 7, 13, 10], scores=[0.744, 0.722, 0.714, 0.708, 0.706]
- **kmeans**: time=0.018028s, top_indices=[6, 8, 7, 13, 10], scores=[1.044, 1.022, 1.014, 1.008, 1.006]
- **hdbscan**: time=0.009491s, top_indices=[6, 8, 7, 13, 10], scores=[1.044, 1.022, 1.014, 1.008, 1.006]

#### Scene Details
| Scene Index | Timestamp | Description |
|---|---|---|
| 6 | 00:00:26.667 | Scene 0:   The scene shows the beginning stages of the pasta-making tutorial. The visuals depict an old kitchen mixer with a measuring cup on top, sug |
| 7 | 00:00:31.833 | <short scene paragraph>   The scene begins with a focus on the pasta-making process. The visuals show flour being used on a tabletop, with a person’s |
| 8 | 00:00:36.000 | <short scene paragraph>   The scene focuses on the early stages of pasta preparation, showing a person working with flour on a wooden table. The visua |
| 10 | 00:00:45.000 | <short scene paragraph>   The scene continues the pasta-making tutorial, focusing on the process of combining eggs and flour. The visuals show flour o |
| 13 | 00:00:56.833 | <short scene paragraph>   The scene continues the pasta-making tutorial, focusing on the combination of ingredients to form dough. The visuals show fl |

### Query: when they serve pasta
- **flat**: time=0.000174s, top_indices=[3, 43, 2, 50, 49], scores=[0.636, 0.632, 0.632, 0.632, 0.631]
- **kmeans**: time=0.008695s, top_indices=[3, 43, 2, 50, 49], scores=[0.936, 0.932, 0.932, 0.932, 0.931]
- **hdbscan**: time=0.009068s, top_indices=[2, 49, 45, 51, 46], scores=[0.932, 0.931, 0.915, 0.906, 0.901]

#### Scene Details
| Scene Index | Timestamp | Description |
|---|---|---|
| 2 | 00:00:11.667 | <short scene paragraph>   The scene depicts the final stages of the pasta-making tutorial, transitioning from preparation to presentation. The visuals |
| 3 | 00:00:14.333 | <short scene paragraph>   The scene transitions to a focus on the completed dish, showcasing a bowl of pasta with cheese and tomatoes. The visuals bri |
| 43 | 00:04:04.000 | The scene shows pasta being cooked in a pan of tomato sauce, with the narrator mentioning, "I'm serving mine with my simple tomato sauce which is supe |
| 45 | 00:04:11.500 | Scene Report:  The scene shows pasta being cooked in a skillet with tomato sauce on a gas stove. The audio contains faint music but provides no explic |
| 46 | 00:04:16.333 | The scene shows someone cooking pasta on a stove, with food being stirred into a pot. The visuals suggest the pasta dish is being combined or finished |
| 49 | 00:04:26.333 | Scene Report:  The scene shows a bowl of pasta and a piece of cheese on a table, followed by someone cutting food with a knife and fork, and then peel |
| 50 | 00:04:29.500 | The scene shows a bowl of pasta garnished with cheese on a table, with the narrator's voice mentioning "some olive oil. There you go." The visuals sug |
| 51 | 00:04:32.167 | The scene shows a bowl of pasta being garnished with cheese while the narrator provides practical advice on making pasta at home, emphasizing its simp |

### Query: a cooking demonstration
- **flat**: time=0.000430s, top_indices=[45, 0, 58, 36, 2], scores=[0.629, 0.62, 0.619, 0.618, 0.617]
- **kmeans**: time=0.010626s, top_indices=[45, 2, 58, 62, 46], scores=[0.929, 0.917, 0.917, 0.915, 0.915]
- **hdbscan**: time=0.009987s, top_indices=[45, 0, 2, 36, 46], scores=[0.927, 0.918, 0.917, 0.917, 0.914]

#### Scene Details
| Scene Index | Timestamp | Description |
|---|---|---|
| 0 | 00:00:00.000 | <short scene paragraph>  The scene depicts the preparation of homemade pasta dough on a cutting board. The video begins with a visual of raw pasta bei |
| 2 | 00:00:11.667 | <short scene paragraph>   The scene depicts the final stages of the pasta-making tutorial, transitioning from preparation to presentation. The visuals |
| 36 | 00:03:23.500 | Scene Report:  The scene shows a person rolling out pasta dough on a wooden table, continuing the hands-on process of preparing it for cutting. The au |
| 45 | 00:04:11.500 | Scene Report:  The scene shows pasta being cooked in a skillet with tomato sauce on a gas stove. The audio contains faint music but provides no explic |
| 46 | 00:04:16.333 | The scene shows someone cooking pasta on a stove, with food being stirred into a pot. The visuals suggest the pasta dish is being combined or finished |

