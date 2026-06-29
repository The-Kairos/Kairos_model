# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 16:16:14 UTC | 3M7s6SupWyU_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 153.823 | 0.656 | 47.715 | 20.040 | 7.914 | 26.387 | 3.380 |
| 2026-06-24 10:12:29 UTC | 3M7s6SupWyU_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 158.835 | 0.665 | 47.553 | 17.638 | 20.369 | 21.945 | 3.387 |

## 2026-06-23 16:16:14 UTC | 3M7s6SupWyU_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3M7s6SupWyU_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `153.823` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.656 |
| save_clips | - |
| sample_frames | 0.895 |
| caption_frames | 35.230 |
| sample_fps | 2.016 |
| detect_object_yolo | 8.206 |
| audio_scan | 9.534 |
| asr_timings | 11.628 |
| ast_timings | 26.544 |
| describe_scenes | 20.040 |
| summarize_scenes | 7.914 |
| synthesize_synopsis | 26.387 |
| make_embedding | 3.380 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.131 |
| branch_yolo_total | 10.228 |
| branch_audio_total | 47.715 |

## 2026-06-24 10:12:29 UTC | 3M7s6SupWyU_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3M7s6SupWyU_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `158.835` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.665 |
| save_clips | - |
| sample_frames | 0.891 |
| caption_frames | 34.607 |
| sample_fps | 2.051 |
| detect_object_yolo | 8.343 |
| audio_scan | 9.629 |
| asr_timings | 11.269 |
| ast_timings | 26.647 |
| describe_scenes | 17.638 |
| summarize_scenes | 20.369 |
| synthesize_synopsis | 21.945 |
| make_embedding | 3.387 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.504 |
| branch_yolo_total | 10.400 |
| branch_audio_total | 47.553 |
