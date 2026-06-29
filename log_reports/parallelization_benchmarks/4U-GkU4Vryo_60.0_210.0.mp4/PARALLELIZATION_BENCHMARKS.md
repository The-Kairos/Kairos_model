# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 17:04:45 UTC | 4U-GkU4Vryo_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 182.627 | 0.667 | 70.393 | 19.391 | 17.187 | 23.055 | 3.323 |
| 2026-06-24 10:58:42 UTC | 4U-GkU4Vryo_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 170.787 | 0.678 | 61.348 | 20.999 | 19.960 | 14.429 | 3.267 |

## 2026-06-23 17:04:45 UTC | 4U-GkU4Vryo_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4U-GkU4Vryo_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `182.627` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.667 |
| save_clips | - |
| sample_frames | 1.002 |
| caption_frames | 35.752 |
| sample_fps | 2.052 |
| detect_object_yolo | 8.442 |
| audio_scan | 12.630 |
| asr_timings | 31.204 |
| ast_timings | 26.549 |
| describe_scenes | 19.391 |
| summarize_scenes | 17.187 |
| synthesize_synopsis | 23.055 |
| make_embedding | 3.323 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.760 |
| branch_yolo_total | 10.500 |
| branch_audio_total | 70.393 |

## 2026-06-24 10:58:42 UTC | 4U-GkU4Vryo_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4U-GkU4Vryo_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `170.787` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.678 |
| save_clips | - |
| sample_frames | 0.996 |
| caption_frames | 37.075 |
| sample_fps | 2.062 |
| detect_object_yolo | 8.603 |
| audio_scan | 12.820 |
| asr_timings | 21.767 |
| ast_timings | 26.751 |
| describe_scenes | 20.999 |
| summarize_scenes | 19.960 |
| synthesize_synopsis | 14.429 |
| make_embedding | 3.267 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.077 |
| branch_yolo_total | 10.671 |
| branch_audio_total | 61.348 |
