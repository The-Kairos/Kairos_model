# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 00:47:24 UTC | uErmi6kKVIs_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 144.253 | 0.798 | 53.208 | 13.657 | 6.144 | 10.319 | 3.566 |

## 2026-06-27 00:47:24 UTC | uErmi6kKVIs_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/uErmi6kKVIs_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `144.253` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.798 |
| save_clips | - |
| sample_frames | 1.448 |
| caption_frames | 42.044 |
| sample_fps | 2.405 |
| detect_object_yolo | 9.165 |
| audio_scan | 14.010 |
| asr_timings | 8.201 |
| ast_timings | 30.989 |
| describe_scenes | 13.657 |
| summarize_scenes | 6.144 |
| synthesize_synopsis | 10.319 |
| make_embedding | 3.566 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.498 |
| branch_yolo_total | 11.575 |
| branch_audio_total | 53.208 |
