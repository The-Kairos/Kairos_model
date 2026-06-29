# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 03:47:47 UTC | Hdfmhdj3Efg_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 98.527 | 0.934 | 35.547 | 5.355 | 8.917 | 11.794 | 2.379 |

## 2026-06-25 03:47:47 UTC | Hdfmhdj3Efg_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Hdfmhdj3Efg_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `98.527` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.934 |
| save_clips | - |
| sample_frames | 0.925 |
| caption_frames | 22.235 |
| sample_fps | 2.088 |
| detect_object_yolo | 6.959 |
| audio_scan | 12.739 |
| asr_timings | 6.875 |
| ast_timings | 15.924 |
| describe_scenes | 5.355 |
| summarize_scenes | 8.917 |
| synthesize_synopsis | 11.794 |
| make_embedding | 2.379 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.165 |
| branch_yolo_total | 9.053 |
| branch_audio_total | 35.547 |
