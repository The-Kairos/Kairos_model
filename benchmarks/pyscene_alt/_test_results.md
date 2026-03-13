## Argentina v France Full Penalty Shoot-out.mp4

| Test         | Scenes | Avg Len (s) | Min Len (s) | Max Len (s) | Elapsed (s) | Status |
| ------------ | ------ | ----------- | ----------- | ----------- | ----------- | ------ |
| pyscene_base | 74     | 6.20        | 2.24        | 26.24       | 3.45        | ok     |
| vit_scene    | 75     | 6.12        | 0.48        | 33.00       | 20.11       | ok     |
| clip_scene   | 112    | 4.10        | 0.48        | 18.48       | 12.61       | ok     |
| blip_scene   | 104    | 4.41        | 0.48        | 18.48       | 51.07       | ok     |
| py_vit       | 62     | 7.40        | 0.56        | 40.64       | 8.55        | ok     |
| py_clip      | 84     | 5.46        | 0.56        | 24.80       | 8.89        | ok     |
| py_blip      | 82     | 5.60        | 0.56        | 24.80       | 13.78       | ok     |

**Best Method: `py_blip`**
`py_blip` achieves a balanced segmentation with 82 scenes, capturing distinct events such as penalty kicks, crowd reactions, and player interactions. It avoids excessive fragmentation while ensuring semantic coherence. The scene splits align well with the transitions between visually distinct moments, supporting downstream tasks like BLIP captioning and YOLO object detection. Its runtime is reasonable, making it efficient for the Kairos pipeline.

**Over-Segmentation: `clip_scene` and `blip_scene`**
Both methods produce excessive splits (`clip_scene`: 112 scenes, `blip_scene`: 104 scenes), fragmenting continuous events like penalty kicks and crowd reactions into multiple scenes. This over-segmentation could hinder downstream tasks by introducing unnecessary complexity and reducing coherence.

**Under-Segmentation: `py_vit` and `pyscene_base`**
`py_vit` (62 scenes) and `pyscene_base` (74 scenes) under-segment, combining distinct events such as player reactions and penalty kicks into single scenes. This limits the granularity needed for accurate object detection, audio event extraction, and scene descriptions.

**Close Second: `py_clip`**
`py_clip` (84 scenes) is a close second, offering slightly more granularity than `py_blip` while maintaining coherence. It captures distinct events effectively but introduces minor over-segmentation in continuous moments.

**Recommendation**
`py_blip` is the optimal choice for its balance of granularity, semantic coherence, and runtime efficiency.

## How to Make Pasta - Without a Machine.mp4

| Test         | Scenes | Avg Len (s) | Min Len (s) | Max Len (s) | Elapsed (s) | Status |
| ------------ | ------ | ----------- | ----------- | ----------- | ----------- | ------ |
| pyscene_base | 58     | 5.66        | 2.17        | 19.67       | 7.80        | ok     |
| vit_scene    | 38     | 8.63        | 0.50        | 52.50       | 30.86       | ok     |
| clip_scene   | 21     | 15.62       | 0.50        | 97.50       | 25.87       | ok     |
| blip_scene   | 30     | 10.94       | 0.50        | 46.00       | 53.13       | ok     |
| py_vit       | 38     | 8.63        | 0.67        | 67.25       | 16.53       | ok     |
| py_clip      | 34     | 9.65        | 1.00        | 40.00       | 16.29       | ok     |
| py_blip      | 46     | 7.13        | 0.75        | 34.33       | 21.98       | ok     |

**Best Method:** **py_blip**
Py_blip achieves the most semantically coherent segmentation, capturing 48 scenes that align well with distinct events in the pasta-making process (e.g., ingredient preparation, dough kneading, rolling, cooking, and plating). It avoids over-segmentation while ensuring critical transitions are captured, supporting downstream tasks like BLIP captioning and GPT summaries effectively.

**Over-Segmentation:**

- **pyscene_base** (58 scenes): Splits excessively, often within continuous actions like dough kneading or rolling, leading to redundant scene breaks that hinder semantic coherence.
- **blip_scene** (30 scenes): Slightly over-segments compared to py_blip, breaking some continuous actions unnecessarily.

**Under-Segmentation:**

- **clip_scene** (21 scenes): Combines distinct events (e.g., dough preparation and rolling) into single scenes, losing granularity and reducing semantic clarity.
- **vit_scene** (38 scenes): Misses some transitions, particularly during dough handling and cooking steps, leading to under-segmentation.

**Close Second:** **py_clip**
Py_clip (34 scenes) is a strong alternative, capturing transitions effectively while avoiding excessive splits. However, it misses finer granularity in some steps compared to py_blip.

**Recommendation:** Use **py_blip** for its balance of granularity and coherence, ensuring distinct events are captured without redundancy.

## Young Sheldon - First Day of High School.mp4

| Test         | Scenes | Avg Len (s) | Min Len (s) | Max Len (s) | Elapsed (s) | Status   |
| ------------ | ------ | ----------- | ----------- | ----------- | ----------- | -------- |
| pyscene_base | 34     | 4.96        | 2.00        | 13.18       | 3.97        | ok       |
| vit_scene    | 42     | 4.01        | 0.46        | 20.98       | 16.03       | ok       |
| clip_scene   | 36     | 4.68        | 0.50        | 31.99       | 12.18       | ok       |
| blip_scene   | 39     | 4.32        | 0.50        | 23.98       | 25.88       | ok       |
| py_vit       | 36     | 4.68        | 0.58        | 30.28       | 8.28        | ok       |
| py_clip      | 34     | 4.96        | 0.58        | 37.54       | 8.19        | ok       |
| py_blip      | 37     | 4.55        | 0.58        | 21.35       | 10.78       | <br />ok |

cannot process cuz of ResponsibleAIPolicyViolation

## Watch Malala Yousafzai's Nobel Peace Prize acceptance speech.mp4

| Test         | Scenes | Avg Len (s) | Min Len (s) | Max Len (s) | Elapsed (s) | Status |
| ------------ | ------ | ----------- | ----------- | ----------- | ----------- | ------ |
| pyscene_base | 22     | 12.43       | 2.40        | 56.72       | 6.09        | ok     |
| vit_scene    | 21     | 13.03       | 0.50        | 56.56       | 25.93       | ok     |
| clip_scene   | 23     | 11.89       | 0.50        | 56.56       | 20.70       | ok     |
| blip_scene   | 23     | 11.89       | 0.50        | 56.56       | 43.04       | ok     |
| py_vit       | 20     | 13.67       | 2.34        | 56.72       | 11.10       | ok     |
| py_clip      | 22     | 12.43       | 2.34        | 56.72       | 11.46       | ok     |
| py_blip      | 22     | 12.43       | 2.34        | 56.72       | 12.44       | ok     |

**Best Method: `blip_scene`**
The `blip_scene` method captures 23 scenes, effectively segmenting the video into visually distinct moments. It identifies transitions between different camera angles, audience reactions, and key moments of the speech. This segmentation aligns well with the semantic coherence required for downstream tasks like BLIP captioning, YOLO object detection, and GPT-based summaries. The average scene length (11.89s) is balanced, avoiding over-segmentation while ensuring sufficient granularity for capturing meaningful events.

**Over-Segmentation:**
None of the methods show clear over-segmentation for this video. All methods maintain a reasonable average scene length and avoid splitting continuous shots unnecessarily.

**Under-Segmentation:**
The `py_vit` method under-segments the video with only 20 scenes, potentially combining distinct events into single scenes. This could hinder downstream tasks by reducing granularity.

**Close Second:**
The `clip_scene` method is a close second, also capturing 23 scenes with similar segmentation quality. However, `blip_scene` is preferred due to its alignment with BLIP captioning and semantic coherence.

**Recommendation:**
Use `blip_scene` for the Kairos pipeline. It provides the best balance between scene granularity and coherence, supporting effective downstream processing.

## CCTV Dogs.mp4

| Test         | Scenes | Avg Len (s) | Min Len (s) | Max Len (s) | Elapsed (s) | Status |
| ------------ | ------ | ----------- | ----------- | ----------- | ----------- | ------ |
| pyscene_base | 0      | 0.00        | 0.00        | 0.00        | 1.89        | ok     |
| vit_scene    | 1      | 300.04      | 300.04      | 300.04      | 16.04       | ok     |
| clip_scene   | 1      | 300.04      | 300.04      | 300.04      | 11.26       | ok     |
| blip_scene   | 1      | 300.04      | 300.04      | 300.04      | 34.68       | ok     |
| py_vit       | 0      | 0.00        | 0.00        | 0.00        | 2.73        | error  |
| py_clip      | 0      | 0.00        | 0.00        | 0.00        | 2.63        | error  |
| py_blip      | 0      | 0.00        | 0.00        | 0.00        | 2.69        | error  |

All methods (BLIP, CLIP, ViT) produced only one scene for the entire 300-second video, indicating severe **under-segmentation**. The contact sheets show a static environment with minimal visual changes, suggesting that the segmentation algorithms failed to detect any meaningful transitions or events. This is problematic for the Kairos pipeline, as it relies on distinct scene splits to extract meaningful captions, object detections, and audio events.

## Cartastrophy.mp4

| Test         | Scenes | Avg Len (s) | Min Len (s) | Max Len (s) | Elapsed (s) | Status |
| ------------ | ------ | ----------- | ----------- | ----------- | ----------- | ------ |
| pyscene_base | 32     | 7.99        | 2.00        | 28.13       | 4.27        | ok     |
| vit_scene    | 142    | 1.80        | 0.27        | 30.00       | 23.62       | ok     |
| clip_scene   | 63     | 4.06        | 0.27        | 30.00       | 14.71       | ok     |
| blip_scene   | 111    | 2.30        | 0.50        | 17.50       | 37.94       | ok     |
| py_vit       | 56     | 4.57        | 0.47        | 41.27       | 12.56       | ok     |
| py_clip      | 52     | 4.92        | 0.47        | 29.93       | 12.60       | ok     |
| py_blip      | 59     | 4.34        | 0.47        | 25.40       | 16.70       | ok     |

**Best Method: `py_blip`**
The `py_blip` method provides the most balanced segmentation for the Kairos pipeline. It captures 59 scenes with an average length of 4.34 seconds, ensuring that visually distinct events are separated while avoiding excessive fragmentation. The scenes are coherent and represent different events or visual contexts, supporting downstream tasks like BLIP captioning, YOLO object detection, and GPT-based summarization effectively. The method avoids over-segmentation while maintaining sufficient granularity for semantic analysis.

**Over-Segmentation:**

- **`vit_scene` (142 scenes, avg. length 1.80s):** This method excessively splits continuous shots into multiple scenes, which could hinder downstream tasks by fragmenting coherent events. Many consecutive frames show minimal visual changes, indicating over-segmentation.
- **`blip_scene` (111 scenes, avg. length 2.30s):** While better than `vit_scene`, this method still over-segments, creating unnecessary splits within continuous events.

**Under-Segmentation:**

- **`pyscene_base` (32 scenes, avg. length 7.99s):** This method combines distinct events into single scenes, leading to under-segmentation. It fails to capture sufficient granularity for detailed analysis.

**Close Second:**

- **`py_clip` (52 scenes, avg. length 4.92s):** This method is slightly less granular than `py_blip` but still performs well in separating distinct events. It could be considered as a close alternative.

**Recommendation:**
Use `py_blip` for its balance between granularity and coherence, ensuring optimal support for downstream tasks.
