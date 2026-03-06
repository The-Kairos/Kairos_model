# Retrieval Comparison: Young Sheldon - First Day of High School.mp4
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
| flat | 0.000219 | 0.000 |  |
| kmeans | 0.007735 | 0.933 |  |
| hdbscan | 0.006353 | 0.266 |  |

## Per-query details
### Query: Give me the clip of the boy entering his class
- **flat**: time=0.000286s, top_indices=[26, 22, 34, 25, 21], scores=[0.726, 0.714, 0.71, 0.704, 0.697]
- **kmeans**: time=0.006419s, top_indices=[26, 22, 25, 34, 21], scores=[1.026, 1.014, 1.004, 1.001, 0.997]
- **hdbscan**: time=0.007809s, top_indices=[26, 22, 21, 14, 15], scores=[1.026, 1.014, 0.997, 0.988, 0.983]

#### Scene Details
| Scene Index | Timestamp | Description |
|---|---|---|
| 14 | 00:01:04.731 | <short scene paragraph>   The scene shows a boy and a girl walking toward a school bus, followed by a moment where a woman kisses the boy at the bus s |
| 15 | 00:01:09.069 | The scene shows a boy and a girl standing in front of a school bus, with the boy appearing to look at a woman inside the bus. The setting suggests the |
| 21 | 00:01:34.928 | This scene shows a woman and a boy walking down a hallway, with other people in the background. The setting suggests it is a public space, possibly re |
| 22 | 00:01:39.099 | This scene shows a boy walking down a hallway accompanied by a woman and a man, with other people in the background. The boy appears to be starting hi |
| 25 | 00:01:55.782 | This scene shows a boy walking down a school hallway, passing a red door with the word "ROY" on it. A girl is also seen walking down the hall holding |
| 26 | 00:02:01.621 | This scene shows a boy and a woman standing in a school hallway near red lockers. The woman says, "This is your homeroom. Do you want me to go in with |

### Query: Give me the scene of the mom worried
- **flat**: time=0.000238s, top_indices=[9, 16, 18, 26, 22], scores=[0.732, 0.705, 0.685, 0.683, 0.683]
- **kmeans**: time=0.013017s, top_indices=[16, 18, 26, 22, 28], scores=[1.005, 0.985, 0.983, 0.983, 0.982]
- **hdbscan**: time=0.006133s, top_indices=[16, 18, 19, 17, 14], scores=[1.005, 0.985, 0.979, 0.976, 0.976]

#### Scene Details
| Scene Index | Timestamp | Description |
|---|---|---|
| 9 | 00:00:32.199 | Scene Report:  <short scene paragraph>   The scene shows a woman driving a car with two children seated behind her. The audio features the woman sayin |
| 14 | 00:01:04.731 | <short scene paragraph>   The scene shows a boy and a girl walking toward a school bus, followed by a moment where a woman kisses the boy at the bus s |
| 16 | 00:01:11.238 | The scene shows a boy with a sad expression standing near a woman who appears to be holding his arm. The setting transitions to a boy and a girl looki |
| 17 | 00:01:13.240 | The scene shows a boy with a serious expression standing next to a woman holding a bag, followed by the two standing in front of a school bus. The boy |
| 18 | 00:01:15.742 | <short scene paragraph>   The scene shows a woman speaking to a boy on the street, followed by a girl and boy eating at an outdoor restaurant, and the |
| 19 | 00:01:20.747 | <short scene paragraph>   The scene shows a boy and a girl standing near a school bus, with a woman nearby. The boy appears apprehensive, possibly due |
| 22 | 00:01:39.099 | This scene shows a boy walking down a hallway accompanied by a woman and a man, with other people in the background. The boy appears to be starting hi |
| 26 | 00:02:01.621 | This scene shows a boy and a woman standing in a school hallway near red lockers. The woman says, "This is your homeroom. Do you want me to go in with |
| 28 | 00:02:07.794 | This scene shows Sheldon and his mother standing in a school hallway. His mother says, "Okay, well, you have a good day and I'll pick you up after sch |

### Query: Show me the scenes that have music
- **flat**: time=0.000156s, top_indices=[33, 0, 2, 19, 1], scores=[0.695, 0.688, 0.682, 0.663, 0.658]
- **kmeans**: time=0.005229s, top_indices=[33, 0, 2, 1, 19], scores=[0.995, 0.988, 0.982, 0.958, 0.951]
- **hdbscan**: time=0.005700s, top_indices=[19, 17, 18, 32, 21], scores=[0.963, 0.939, 0.929, 0.928, 0.928]

#### Scene Details
| Scene Index | Timestamp | Description |
|---|---|---|
| 0 | 00:00:00.000 | <short scene paragraph>  The scene depicts a car driving through a rural area, with trees and a park visible in the background. The audio features mus |
| 1 | 00:00:02.836 | A car drives down a street lined with trees and parked cars, followed by a van moving near a parking lot. A boy is seen adjusting his black tie, sugge |
| 2 | 00:00:06.340 | The scene shows a woman driving a car through a rural area. The visuals suggest a calm setting with no significant character interaction or dialogue. |
| 17 | 00:01:13.240 | The scene shows a boy with a serious expression standing next to a woman holding a bag, followed by the two standing in front of a school bus. The boy |
| 18 | 00:01:15.742 | <short scene paragraph>   The scene shows a woman speaking to a boy on the street, followed by a girl and boy eating at an outdoor restaurant, and the |
| 19 | 00:01:20.747 | <short scene paragraph>   The scene shows a boy and a girl standing near a school bus, with a woman nearby. The boy appears apprehensive, possibly due |
| 21 | 00:01:34.928 | This scene shows a woman and a boy walking down a hallway, with other people in the background. The setting suggests it is a public space, possibly re |
| 32 | 00:02:27.147 | This scene shows Sheldon walking down a school hallway, possibly accompanied by a teacher. Another moment depicts a boy and girl looking at each other |
| 33 | 00:02:35.322 | This scene shows a series of abstract visuals, including an empty square with a white rectangle, the "subbie" logo with a white outline on a green bac |

### Query: Show me clips of the school
- **flat**: time=0.000167s, top_indices=[38, 37, 22, 34, 20], scores=[0.649, 0.648, 0.646, 0.643, 0.639]
- **kmeans**: time=0.006592s, top_indices=[38, 37, 22, 34, 20], scores=[0.949, 0.947, 0.946, 0.942, 0.939]
- **hdbscan**: time=0.005970s, top_indices=[22, 26, 21, 32, 17], scores=[0.946, 0.937, 0.936, 0.935, 0.935]

#### Scene Details
| Scene Index | Timestamp | Description |
|---|---|---|
| 17 | 00:01:13.240 | The scene shows a boy with a serious expression standing next to a woman holding a bag, followed by the two standing in front of a school bus. The boy |
| 20 | 00:01:24.418 | <short scene paragraph>   The scene shows a group of young people walking down a school hallway, with one holding an open book. The setting suggests i |
| 21 | 00:01:34.928 | This scene shows a woman and a boy walking down a hallway, with other people in the background. The setting suggests it is a public space, possibly re |
| 22 | 00:01:39.099 | This scene shows a boy walking down a hallway accompanied by a woman and a man, with other people in the background. The boy appears to be starting hi |
| 26 | 00:02:01.621 | This scene shows a boy and a woman standing in a school hallway near red lockers. The woman says, "This is your homeroom. Do you want me to go in with |
| 32 | 00:02:27.147 | This scene shows Sheldon walking down a school hallway, possibly accompanied by a teacher. Another moment depicts a boy and girl looking at each other |

### Query: A clip where students are in a classroom
- **flat**: time=0.000248s, top_indices=[26, 20, 25, 22, 15], scores=[0.697, 0.688, 0.687, 0.684, 0.683]
- **kmeans**: time=0.007420s, top_indices=[26, 20, 25, 22, 15], scores=[0.997, 0.988, 0.987, 0.984, 0.983]
- **hdbscan**: time=0.006152s, top_indices=[26, 22, 15, 21, 31], scores=[0.997, 0.984, 0.981, 0.977, 0.976]

#### Scene Details
| Scene Index | Timestamp | Description |
|---|---|---|
| 15 | 00:01:09.069 | The scene shows a boy and a girl standing in front of a school bus, with the boy appearing to look at a woman inside the bus. The setting suggests the |
| 20 | 00:01:24.418 | <short scene paragraph>   The scene shows a group of young people walking down a school hallway, with one holding an open book. The setting suggests i |
| 21 | 00:01:34.928 | This scene shows a woman and a boy walking down a hallway, with other people in the background. The setting suggests it is a public space, possibly re |
| 22 | 00:01:39.099 | This scene shows a boy walking down a hallway accompanied by a woman and a man, with other people in the background. The boy appears to be starting hi |
| 25 | 00:01:55.782 | This scene shows a boy walking down a school hallway, passing a red door with the word "ROY" on it. A girl is also seen walking down the hall holding |
| 26 | 00:02:01.621 | This scene shows a boy and a woman standing in a school hallway near red lockers. The woman says, "This is your homeroom. Do you want me to go in with |
| 31 | 00:02:19.473 | This scene shows Sheldon standing in front of a red school door, likely preparing to enter a classroom. The audio includes Sheldon saying, "It's proba |

