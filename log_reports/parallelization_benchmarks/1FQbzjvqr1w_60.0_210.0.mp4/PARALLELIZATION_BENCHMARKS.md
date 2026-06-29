# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 13:09:38 UTC | 1FQbzjvqr1w_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 202.166 | 0.752 | 78.634 | 31.991 | 13.071 | 20.299 | 3.808 |
| 2026-06-27 14:49:44 UTC | 1FQbzjvqr1w_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 166.398 | 0.789 | 83.303 | 8.749 | 7.281 | 7.800 | 3.659 |

## 2026-06-23 13:09:38 UTC | 1FQbzjvqr1w_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1FQbzjvqr1w_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `202.166` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.752 |
| save_clips | - |
| sample_frames | 1.118 |
| caption_frames | 39.983 |
| sample_fps | 2.231 |
| detect_object_yolo | 8.875 |
| audio_scan | 14.757 |
| asr_timings | 34.073 |
| ast_timings | 29.796 |
| describe_scenes | 31.991 |
| summarize_scenes | 13.071 |
| synthesize_synopsis | 20.299 |
| make_embedding | 3.808 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.107 |
| branch_yolo_total | 11.112 |
| branch_audio_total | 78.634 |

## 2026-06-27 14:49:44 UTC | 1FQbzjvqr1w_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1FQbzjvqr1w_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `166.398` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.789 |
| save_clips | - |
| sample_frames | 1.149 |
| caption_frames | 40.729 |
| sample_fps | 2.269 |
| detect_object_yolo | 9.228 |
| audio_scan | 15.059 |
| asr_timings | 38.281 |
| ast_timings | 29.956 |
| describe_scenes | 8.749 |
| summarize_scenes | 7.281 |
| synthesize_synopsis | 7.800 |
| make_embedding | 3.659 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.884 |
| branch_yolo_total | 11.503 |
| branch_audio_total | 83.303 |
