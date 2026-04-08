# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-07 20:39:57 UTC | SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source.mp4 | semi_parallel | gemini | gemini-embedding-001 | 2.799 | - | - | - | - | - | 0.858 |
| 2026-04-07 21:23:02 UTC | SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source.mp4 | semi_parallel | gemini | gemini-embedding-001 | 1.458 | - | - | - | - | - | 0.858 |
| 2026-04-08 07:11:33 UTC | SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source.mp4 | semi_parallel | gemini | gemini-embedding-001 | 63.642 | 0.309 | 34.813 | 7.964 | 6.931 | 7.942 | 0.934 |
| 2026-04-08 07:15:10 UTC | SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source.mp4 | semi_parallel | gemini | gemini-embedding-001 | 56.408 | 0.197 | 26.074 | 9.967 | 7.969 | 7.027 | 0.737 |

## 2026-04-07 20:39:57 UTC | SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source.mp4 | semi_parallel

- Video path: `/var/tmp/kairos/jobs/1d4dec88-794c-4ac1-bde3-0b27cf45802e/SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `2.799` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | - |
| save_clips | - |
| sample_frames | - |
| caption_frames | - |
| sample_fps | - |
| detect_object_yolo | - |
| audio_scan | - |
| asr_timings | - |
| ast_timings | - |
| describe_scenes | - |
| summarize_scenes | - |
| synthesize_synopsis | - |
| make_embedding | 0.858 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | - |
| branch_yolo_total | - |
| branch_audio_total | - |

## 2026-04-07 21:23:02 UTC | SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source.mp4 | semi_parallel

- Video path: `/var/tmp/kairos/jobs/50d175d9-830c-48fe-91c5-d8bc682febc8/SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `1.458` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | - |
| save_clips | - |
| sample_frames | - |
| caption_frames | - |
| sample_fps | - |
| detect_object_yolo | - |
| audio_scan | - |
| asr_timings | - |
| ast_timings | - |
| describe_scenes | - |
| summarize_scenes | - |
| synthesize_synopsis | - |
| make_embedding | 0.858 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | - |
| branch_yolo_total | - |
| branch_audio_total | - |

## 2026-04-08 07:11:33 UTC | SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source.mp4 | semi_parallel

- Video path: `/var/tmp/kairos/jobs/58b13e6f-8035-4be8-bd4e-2b755e821fc5/SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `63.642` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.309 |
| save_clips | - |
| sample_frames | 0.297 |
| caption_frames | 11.161 |
| sample_fps | 0.358 |
| detect_object_yolo | 1.950 |
| audio_scan | 7.600 |
| asr_timings | 3.921 |
| ast_timings | 9.508 |
| describe_scenes | 7.964 |
| summarize_scenes | 6.931 |
| synthesize_synopsis | 7.942 |
| make_embedding | 0.934 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 11.463 |
| branch_yolo_total | 2.312 |
| branch_audio_total | 21.037 |

## 2026-04-08 07:15:10 UTC | SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source.mp4 | semi_parallel

- Video path: `/var/tmp/kairos/jobs/ba34935c-6761-4c48-bb53-3748b311c6bc/SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `56.408` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.197 |
| save_clips | - |
| sample_frames | 0.296 |
| caption_frames | 4.023 |
| sample_fps | 0.356 |
| detect_object_yolo | 1.911 |
| audio_scan | 6.354 |
| asr_timings | 3.599 |
| ast_timings | 9.518 |
| describe_scenes | 9.967 |
| summarize_scenes | 7.969 |
| synthesize_synopsis | 7.027 |
| make_embedding | 0.737 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 4.323 |
| branch_yolo_total | 2.272 |
| branch_audio_total | 19.478 |
