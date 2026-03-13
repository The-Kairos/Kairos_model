# Processing Logs Summary

## [WithOUT Yolo] How to Make Pasta - Without a Machine.mp4

**Summary:**  
The video is a step-by-step cooking demonstration focused on making homemade pasta dough in a rustic kitchen setting. The process starts with introducing flour, eggs, olive oil, and salt as the key ingredients, followed by methodically explaining how to combine and knead the dough to achieve proper texture and elasticity. After resting the dough, it is divided, rolled thin, folded, and cut into tagliatelle strips. Tips are offered for storing and cooking fresh pasta, which is then briefly boiled and combined with homemade tomato sauce. The completed dish is garnished with arugula, Parmesan, and olive oil, suggesting bread as an accompaniment. The presenter emphasizes the simplicity, practicality, and accessibility of making pasta at home, particularly when store-bought options are unavailable. The video ends by encouraging viewers to continue exploring pasta recipes, engage through social media and comments, and subscribe for future content, maintaining a calm, approachable tone throughout.

|  wall_time_%  | step                             |   wall_time_sec |   cpu_time_sec |   ram_used_MB |   io_read_MB |   io_write_MB |
|:-------------:|:---------------------------------|----------------:|---------------:|--------------:|-------------:|--------------:|
|     0.3%      | PySceneDetect*                   |           0.996 |          4.779 |         2.561 |        65.91 |             0 |
|     9.3%      | AST sound descriptions*          |          29.813 |         29.691 |        67.866 |       66.389 |             0 |
|   **34.8%**   | ASR speech transcription*        |         111.506 |         111.26 |         40.61 |      234.309 |             0 |
|     0.4%      | Masked clips saving              |           0.108 |          0.005 |             0 |        0.003 |             0 |
|     0.4%      | Frame sampling                   |           0.125 |          0.954 |         0.649 |       11.843 |         0.063 |
|     25.8%     | BLIP caption                     |           7.923 |          7.899 |        17.825 |        0.001 |             0 |
|     0.0%      | YOLO detection*                  |               0 |              0 |         0.012 |        0.067 |             0 |
|     25.3%     | BLIP + YOLO + AST + ASR in GPT4o |           2.778 |          0.013 |        -0.105 |        0.022 |             0 |
|     2.8%      | Summarization*                   |           0.149 |              0 |             0 |            0 |             0 |
|     0.8%      | Synopsis + common Q&A*           |           0.044 |              0 |             0 |            0 |             0 |

**Footnote:**  
`total_process_sec` without LLM cooldown (5.00s per scene, 1464.67s total) is **4.47x longer** than `video_length` of 328.00s.
**57.0 scenes** were detected in `Videos\[WithOUT Yolo] How to Make Pasta - Without a Machine.mp4`
\* measured per minute of video, whereas the remaining processes are measured per scenes.
