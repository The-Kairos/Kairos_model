# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 12:03:57 UTC | 5W8Wc_u78VU_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 178.003 | 0.656 | 60.658 | 15.024 | 17.785 | 20.031 | 4.165 |

## 2026-06-24 12:03:57 UTC | 5W8Wc_u78VU_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/5W8Wc_u78VU_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `178.003` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.656 |
| save_clips | - |
| sample_frames | 1.181 |
| caption_frames | 45.317 |
| sample_fps | 2.189 |
| detect_object_yolo | 9.595 |
| audio_scan | 14.891 |
| asr_timings | 11.047 |
| ast_timings | 34.712 |
| describe_scenes | 15.024 |
| summarize_scenes | 17.785 |
| synthesize_synopsis | 20.031 |
| make_embedding | 4.165 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.503 |
| branch_yolo_total | 11.789 |
| branch_audio_total | 60.658 |
