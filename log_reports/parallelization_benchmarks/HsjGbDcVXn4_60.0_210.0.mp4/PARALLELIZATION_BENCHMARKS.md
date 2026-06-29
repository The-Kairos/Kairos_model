# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 03:56:35 UTC | HsjGbDcVXn4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 90.999 | 0.796 | 26.804 | 8.863 | 6.707 | 11.827 | 2.538 |

## 2026-06-25 03:56:35 UTC | HsjGbDcVXn4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/HsjGbDcVXn4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `90.999` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.796 |
| save_clips | - |
| sample_frames | 0.765 |
| caption_frames | 26.033 |
| sample_fps | 2.053 |
| detect_object_yolo | 7.468 |
| audio_scan | 3.831 |
| asr_timings | 0.000 |
| ast_timings | 18.716 |
| describe_scenes | 8.863 |
| summarize_scenes | 6.707 |
| synthesize_synopsis | 11.827 |
| make_embedding | 2.538 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.804 |
| branch_yolo_total | 9.527 |
| branch_audio_total | 22.555 |
