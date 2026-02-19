# Processing Logs Summary

## Young Sheldon - First Day of High School.mp4

**Summary:**  
The video begins with a person skateboarding before transitioning to vehicles on a street. A truck moves in an erratic pattern as a young male passenger ("lad") is seen in a car. The focus shifts to a woman driving, suggesting she might play a central role in the narrative. Soon after, the lad, now stationary and wearing a bow tie, is asked, "Would you like to play a driving game?" Moments later, the lad dismisses the idea and follows with reflections on losing games, discussing prime numbers, and intellectual confidence in brief exchanges during the car ride with the woman. As the journey progresses into a social environment at what appears to be a school, the lad interacts with various individuals and a woman believed to be his mother. Themes of individuality, intellect, social dynamics, and familial support thread through scenes of hallway interactions, banter about dress codes, casual observations, and ambiguous dialogue. The video concludes with the lad and others navigating the hallways before shifting to a final abstract scene with a blue-green background and the word "sub," signaling a visual pause or tone change.

|  wall_time_%  | step                             |   wall_time_sec |   cpu_time_sec |   ram_used_MB |   io_read_MB |   io_write_MB |
|:-------------:|:---------------------------------|----------------:|---------------:|--------------:|-------------:|--------------:|
|     0.4%      | PySceneDetect*                   |           1.046 |           4.98 |         4.273 |       57.253 |             0 |
|     5.3%      | AST sound descriptions*          |          13.298 |         99.863 |       141.719 |       58.537 |             0 |
|     18.3%     | ASR speech transcription*        |          45.939 |        348.562 |        37.388 |      385.398 |             0 |
|     0.4%      | Masked clips saving              |           0.074 |          0.005 |             0 |        0.003 |             0 |
|     0.6%      | Frame sampling                   |            0.12 |          1.003 |         0.618 |        9.582 |         0.062 |
|   **43.3%**   | BLIP caption                     |           8.981 |          8.948 |        30.294 |        0.001 |             0 |
|     8.5%      | YOLO detection*                  |           0.355 |          2.793 |        13.062 |         0.13 |         0.097 |
|     13.3%     | BLIP + YOLO + AST + ASR in GPT4o |           2.762 |           0.01 |        -107.5 |        0.001 |             0 |
|     0.0%      | Summarization*                   |               0 |              0 |             0 |            0 |             0 |
|     4.2%      | Synopsis + common Q&A*           |           0.176 |              0 |             0 |            0 |             0 |

**Footnote:**  
`total_process_sec` without LLM cooldown (0.00s per scene, 704.49s total) is **4.18x longer** than `video_length` of 168.50s.
**34.0 scenes** were detected in `Videos\Young Sheldon - First Day of High School.mp4`
\* measured per minute of video, whereas the remaining processes are measured per scenes.
