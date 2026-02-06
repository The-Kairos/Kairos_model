# Processing Logs Summary

(I forgot to run yolo for most of them oops)

## .Malala Yousafzai FULL Nobel Peace Prize Lecture 2014.mp4

Summary: A formal event, likely an award ceremony, unfolds in a structured and respectful atmosphere focused on themes of education, empowerment, and global unity. Multiple speakers deliver speeches emphasizing advocacy for girls' education, systemic challenges, and resilience in the face of adversity, highlighting personal and global stories. Central to the event is a woman wearing a red shawl, later a pink shawl, who passionately recounts her experiences and calls for worldwide commitment to children's education, addressing issues like poverty, child marriage, terrorism, and war. She announces dedicating her Nobel Prize funds to establishing schools in underserved areas, particularly in Pakistan, and challenges the audience and global leaders to act decisively for equal education. Her speeches stress urgency, global solidarity, and optimism, blending heartfelt reflections with occasional humor. The event concludes with applause, expressions of gratitude, and affirmations of shared purpose and action.

| step                             | wall_time_sec | cpu_time_sec | ram_used_MB | io_read_MB | io_write_MB |
| :------------------------------- | ------------: | -----------: | ----------: | ---------: | ----------: |
| PySceneDetect*                   |         0.221 |        1.304 |       1.124 |      3.027 |           0 |
| AST sound descriptions*          |        12.096 |       12.039 |      -1.159 |      3.008 |           0 |
| ASR speech transcription*        |        56.372 |       56.127 |      15.979 |     35.635 |           0 |
| Masked clips saving              |         0.155 |        0.004 |           0 |      0.003 |           0 |
| Frame sampling                   |         0.069 |         0.26 |       0.382 |      1.298 |       0.067 |
| BLIP caption                     |         8.065 |        8.039 |       0.918 |          0 |           0 |
| YOLO detection                   |         0.001 |            0 |         0.2 |      0.199 |           0 |
| BLIP + YOLO + AST + ASR in GPT4o |         2.956 |         0.01 |      -1.118 |      0.012 |           0 |
| Summarization*                   |         0.094 |            0 |       0.001 |          0 |           0 |
| Synopsis + common Q&A*           |         0.007 |            0 |           0 |          0 |           0 |

**Footnote:**
`total_process_sec` without LLM cooldown of 3365.31s is **1.97x longer** than `video_length` of 1708.44s.
**110.0 scenes** were detected in `Videos\.Malala Yousafzai FULL Nobel Peace Prize Lecture 2014.mp4`
\* `get_scene_list`, `ast_timings`,  `asr_timings`, `summarize_scenes`, and `synthesize_synopsis` are measured per minute of video, whereas the remaining processes are measured per scenes.

## Argentina v France Full Penalty Shoot-out.mp4

**Summary:**
The video chronicles a high-stakes soccer match set in an electrified stadium, focusing on pivotal gameplay, emotional reactions, and celebratory moments. It opens with references to past tournaments, commentary providing context, and images of players engaging in critical plays, including goals, missed opportunities, and strategic exchanges. Emotional moments capture the toll of competition, such as a distraught player comforted by their coach. The match builds toward decisive moments, with players making impactful kicks or gestures, ultimately leading to a signature goal that brings Argentina near victory. Joy erupts from both players and spectators, showcasing unified celebrations through hugs, chanting, and flag-waving. Emotional and historic significance is underscored by commentary linking the victory to previous legends. This climaxes with overwhelming triumph for Argentina, celebrated by players falling to their knees, raising hands, and embracing. The video shifts abruptly to abstract imagery featuring lights, an airplane, and text, disconnecting from the stadium’s lively atmosphere.

| step                             | wall_time_sec | cpu_time_sec | ram_used_MB | io_read_MB | io_write_MB |
| :------------------------------- | ------------: | -----------: | ----------: | ---------: | ----------: |
| PySceneDetect*                   |         0.291 |        1.853 |       0.915 |      5.133 |           0 |
| AST sound descriptions*          |        26.068 |        25.98 |     -10.065 |      5.108 |           0 |
| ASR speech transcription*        |       108.306 |      108.066 |      65.752 |    125.721 |           0 |
| Masked clips saving              |         0.078 |        0.003 |           0 |      0.003 |           0 |
| Frame sampling                   |         0.071 |        0.288 |        0.81 |      1.065 |       0.068 |
| BLIP caption                     |         7.568 |        7.545 |      -1.937 |          0 |           0 |
| YOLO detection                   |             0 |        0.001 |       0.013 |      0.277 |           0 |
| BLIP + YOLO + AST + ASR in GPT4o |          2.88 |        0.006 |      -0.203 |      0.001 |           0 |
| Summarization*                   |          0.21 |            0 |           0 |          0 |           0 |
| Synopsis + common Q&A*           |         0.038 |            0 |           0 |          0 |           0 |

**Footnote:**
`total_process_sec` without LLM cooldown of 1981.41s is **4.32x longer** than `video_length` of 459.00s.
**79.0 scenes** were detected in `Videos\Argentina v France Full Penalty Shoot-out.mp4`
\* `get_scene_list`, `ast_timings`,  `asr_timings`, `summarize_scenes`, and `synthesize_synopsis` are measured per minute of video, whereas the remaining processes are measured per scenes.

## [With Yolo] How to Make Pasta - Without a Machine.mp4

**Summary:**
The video is an instructional cooking demonstration on making homemade pasta dough and preparing a simple pasta dish. It begins with ingredients being introduced alongside clear, step-by-step guidance by the narrator in a calm and approachable tone, supported by soft background music. The video extensively covers the process of measuring, mixing, kneading, rolling, and cutting the dough into tagliatelle strips, emphasizing practical techniques and flexibility in adjusting the dough's texture. After resting the dough, it is portioned and prepared for storage or immediate use. Fresh pasta is cooked briefly in boiling water and paired with a tomato sauce, while optional garnishes such as arugula, Parmesan cheese, and olive oil are added to enhance the dish. Complementary items, like bread, are also prepared. The video concludes with a reflective commentary on the value of homemade pasta, encouragement for viewers to try the recipe, and an invitation for audience interaction. A future "chef's studio session" is teased, suggesting creative activities involving pasta dough, ending on a relaxed and inviting note.

| step                             | wall_time_sec | cpu_time_sec | ram_used_MB | io_read_MB | io_write_MB |
| :------------------------------- | ------------: | -----------: | ----------: | ---------: | ----------: |
| PySceneDetect*                   |         0.996 |        4.779 |       2.561 |      65.91 |           0 |
| AST sound descriptions*          |        29.813 |       29.691 |      67.866 |     66.389 |           0 |
| ASR speech transcription*        |       111.506 |       111.26 |       40.61 |    234.309 |           0 |
| Masked clips saving              |         0.108 |        0.005 |           0 |      0.003 |           0 |
| Frame sampling                   |         0.125 |        0.954 |       0.649 |     11.843 |       0.063 |
| BLIP caption                     |         7.923 |        7.899 |      17.825 |      0.001 |           0 |
| YOLO detection                   |         0.237 |        1.869 |       1.877 |      0.387 |       0.067 |
| BLIP + YOLO + AST + ASR in GPT4o |         3.004 |        0.011 |       -0.14 |      0.022 |           0 |
| Summarization*                   |         0.197 |            0 |           0 |          0 |           0 |
| Synopsis + common Q&A*           |         0.061 |            0 |           0 |          0 |           0 |

**Footnote:**
`total_process_sec` without LLM cooldown of 1478.15s is **4.51x longer** than `video_length` of 328.00s.
**57.0 scenes** were detected in `Videos\[With Yolo] How to Make Pasta - Without a Machine.mp4`
\* `get_scene_list`, `ast_timings`,  `asr_timings`, `summarize_scenes`, and `synthesize_synopsis` are measured per minute of video, whereas the remaining processes are measured per scenes.

## [WithOUT Yolo] How to Make Pasta - Without a Machine.mp4

**Summary:**
The video is a step-by-step cooking demonstration focused on making homemade pasta dough in a rustic kitchen setting. The process starts with introducing flour, eggs, olive oil, and salt as the key ingredients, followed by methodically explaining how to combine and knead the dough to achieve proper texture and elasticity. After resting the dough, it is divided, rolled thin, folded, and cut into tagliatelle strips. Tips are offered for storing and cooking fresh pasta, which is then briefly boiled and combined with homemade tomato sauce. The completed dish is garnished with arugula, Parmesan, and olive oil, suggesting bread as an accompaniment. The presenter emphasizes the simplicity, practicality, and accessibility of making pasta at home, particularly when store-bought options are unavailable. The video ends by encouraging viewers to continue exploring pasta recipes, engage through social media and comments, and subscribe for future content, maintaining a calm, approachable tone throughout.

| step                             | wall_time_sec | cpu_time_sec | ram_used_MB | io_read_MB | io_write_MB |
| :------------------------------- | ------------: | -----------: | ----------: | ---------: | ----------: |
| PySceneDetect*                   |         0.996 |        4.779 |       2.561 |      65.91 |           0 |
| AST sound descriptions*          |        29.813 |       29.691 |      67.866 |     66.389 |           0 |
| ASR speech transcription*        |       111.506 |       111.26 |       40.61 |    234.309 |           0 |
| Masked clips saving              |         0.108 |        0.005 |           0 |      0.003 |           0 |
| Frame sampling                   |         0.125 |        0.954 |       0.649 |     11.843 |       0.063 |
| BLIP caption                     |         7.923 |        7.899 |      17.825 |      0.001 |           0 |
| YOLO detection                   |         0.001 |        0.001 |        0.07 |      0.385 |           0 |
| BLIP + YOLO + AST + ASR in GPT4o |         2.778 |        0.013 |      -0.105 |      0.022 |           0 |
| Summarization*                   |         0.149 |            0 |           0 |          0 |           0 |
| Synopsis + common Q&A*           |         0.044 |            0 |           0 |          0 |           0 |

**Footnote:**
`total_process_sec` without LLM cooldown of 1464.67s is **4.47x longer** than `video_length` of 328.00s.
**57.0 scenes** were detected in `Videos\[WithOUT Yolo] How to Make Pasta - Without a Machine.mp4`
\* `get_scene_list`, `ast_timings`,  `asr_timings`, `summarize_scenes`, and `synthesize_synopsis` are measured per minute of video, whereas the remaining processes are measured per scenes.
