# Processing Logs Summary

## Argentina v France Full Penalty Shoot-out.mp4

| step                             |   wall_time_sec |   cpu_time_sec |   ram_used_MB |   io_read_MB |   io_write_MB |
|:---------------------------------|----------------:|---------------:|--------------:|-------------:|--------------:|
| PySceneDetect*                   |           0.291 |          1.853 |         0.915 |        5.133 |         0     |
| AST sound descriptions*          |          26.068 |         25.98  |       -10.065 |        5.108 |         0     |
| ASR speech transcription*        |         108.306 |        108.066 |        65.752 |      125.721 |         0     |
| Masked clips saving              |           0.078 |          0.003 |         0     |        0.003 |         0     |
| Frame sampling                   |           0.071 |          0.288 |         0.81  |        1.065 |         0.068 |
| BLIP caption                     |           7.568 |          7.545 |        -1.937 |        0     |         0     |
| YOLO detection                   |           0     |          0.001 |         0.013 |        0.277 |         0     |
| BLIP + YOLO + AST + ASR in GPT4o |           2.88  |          0.006 |        -0.203 |        0.001 |         0     |
| Summarization*                   |           0.21  |          0     |         0     |        0     |         0     |
| Synopsis + common Q&A*           |           0.038 |          0     |         0     |        0     |         0     |

**Footnote:**  
`total_process_sec` without LLM cooldown of 1981.41s is **4.32x longer** than `video_length` of 459.00s.
**79.0 scenes** were detected in `Videos\Argentina v France Full Penalty Shoot-out.mp4`
\* `get_scene_list`, `ast_timings`,  `asr_timings`, `summarize_scenes`, and `synthesize_synopsis` are measured per minute of video, whereas the remaining processes are measured per scenes. 
## How to Make Pasta - Without a Machine.mp4

| step                             |   wall_time_sec |   cpu_time_sec |   ram_used_MB |   io_read_MB |   io_write_MB |
|:---------------------------------|----------------:|---------------:|--------------:|-------------:|--------------:|
| PySceneDetect*                   |           0.996 |          4.779 |         2.561 |       65.91  |         0     |
| AST sound descriptions*          |          29.813 |         29.691 |        67.866 |       66.389 |         0     |
| ASR speech transcription*        |         111.506 |        111.26  |        40.61  |      234.309 |         0     |
| Masked clips saving              |           0.108 |          0.005 |         0     |        0.003 |         0     |
| Frame sampling                   |           0.125 |          0.954 |         0.649 |       11.843 |         0.063 |
| BLIP caption                     |           7.923 |          7.899 |        17.825 |        0.001 |         0     |
| YOLO detection                   |           0.001 |          0.001 |         0.07  |        0.385 |         0     |
| BLIP + YOLO + AST + ASR in GPT4o |           2.778 |          0.013 |        -0.105 |        0.022 |         0     |
| Summarization*                   |           0.149 |          0     |         0     |        0     |         0     |
| Synopsis + common Q&A*           |           0.044 |          0     |         0     |        0     |         0     |

**Footnote:**  
`total_process_sec` without LLM cooldown of 1464.67s is **4.47x longer** than `video_length` of 328.00s.
**57.0 scenes** were detected in `Videos\How to Make Pasta - Without a Machine.mp4`
\* `get_scene_list`, `ast_timings`,  `asr_timings`, `summarize_scenes`, and `synthesize_synopsis` are measured per minute of video, whereas the remaining processes are measured per scenes. 
