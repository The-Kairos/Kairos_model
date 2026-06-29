# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-10 12:22:17 UTC | SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1___2__clip__7__clip.mp4 | parallel | gemini | gemini-embedding-001 | 47.235 | 0.067 | 6.240 | 12.656 | 5.040 | 18.195 | 0.718 |
| 2026-04-10 18:42:42 UTC | SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1___2__clip__7__clip.mp4 | parallel | gemini | gemini-embedding-001 | 33.389 | 0.074 | 5.551 | 10.321 | 4.698 | 7.596 | 0.698 |
| 2026-04-10 18:57:20 UTC | SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1___2__clip__7__clip.mp4 | parallel | gemini | gemini-embedding-001 | 34.474 | 0.078 | 5.814 | 12.063 | 2.958 | 8.558 | 0.694 |
| 2026-04-11 05:07:17 UTC | SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1___2__clip__7__clip.mp4 | parallel | gemini | gemini-embedding-001 | 28.942 | 0.061 | 5.647 | 7.603 | 2.763 | 7.713 | 0.693 |

## 2026-04-10 12:22:17 UTC | SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1___2__clip__7__clip.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/ece2b6c9-f81a-464f-8b96-9afe378dde65/SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1___2__clip__7__clip.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `47.235` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.067 |
| save_clips | - |
| sample_frames | 0.058 |
| caption_frames | 2.017 |
| sample_fps | 0.079 |
| detect_object_yolo | 0.908 |
| audio_scan | 2.403 |
| asr_timings | 3.828 |
| ast_timings | 3.123 |
| describe_scenes | 12.656 |
| summarize_scenes | 5.040 |
| synthesize_synopsis | 18.195 |
| make_embedding | 0.718 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 2.082 |
| branch_yolo_total | 0.995 |
| branch_audio_total | 6.240 |

## 2026-04-10 18:42:42 UTC | SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1___2__clip__7__clip.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/455d9cb7-70e8-4809-bf2c-996524650074/SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1___2__clip__7__clip.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `33.389` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.074 |
| save_clips | - |
| sample_frames | 0.081 |
| caption_frames | 2.781 |
| sample_fps | 0.067 |
| detect_object_yolo | 1.006 |
| audio_scan | 2.263 |
| asr_timings | 2.537 |
| ast_timings | 3.278 |
| describe_scenes | 10.321 |
| summarize_scenes | 4.698 |
| synthesize_synopsis | 7.596 |
| make_embedding | 0.698 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 2.870 |
| branch_yolo_total | 1.081 |
| branch_audio_total | 5.551 |

## 2026-04-10 18:57:20 UTC | SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1___2__clip__7__clip.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/3f276f6e-a451-4d08-ae51-5d41a274ddc0/SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1___2__clip__7__clip.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `34.474` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.078 |
| save_clips | - |
| sample_frames | 0.062 |
| caption_frames | 1.990 |
| sample_fps | 0.082 |
| detect_object_yolo | 0.960 |
| audio_scan | 2.551 |
| asr_timings | 2.624 |
| ast_timings | 3.253 |
| describe_scenes | 12.063 |
| summarize_scenes | 2.958 |
| synthesize_synopsis | 8.558 |
| make_embedding | 0.694 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 2.058 |
| branch_yolo_total | 1.049 |
| branch_audio_total | 5.814 |

## 2026-04-11 05:07:17 UTC | SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1___2__clip__7__clip.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/2804947e-f8a2-4880-a31c-7b502aafe4b5/SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1___2__clip__7__clip.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `28.942` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.061 |
| save_clips | - |
| sample_frames | 0.052 |
| caption_frames | 2.343 |
| sample_fps | 0.074 |
| detect_object_yolo | 0.942 |
| audio_scan | 2.399 |
| asr_timings | 2.680 |
| ast_timings | 3.240 |
| describe_scenes | 7.603 |
| summarize_scenes | 2.763 |
| synthesize_synopsis | 7.713 |
| make_embedding | 0.693 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 2.402 |
| branch_yolo_total | 1.025 |
| branch_audio_total | 5.647 |
