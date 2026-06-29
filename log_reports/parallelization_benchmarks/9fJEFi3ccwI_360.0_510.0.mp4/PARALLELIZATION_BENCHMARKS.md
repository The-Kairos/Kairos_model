# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 18:28:02 UTC | 9fJEFi3ccwI_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 250.615 | 0.931 | 81.729 | 24.091 | 15.701 | 29.266 | 6.580 |

## 2026-06-24 18:28:02 UTC | 9fJEFi3ccwI_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/9fJEFi3ccwI_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `250.615` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.931 |
| save_clips | - |
| sample_frames | 2.531 |
| caption_frames | 72.167 |
| sample_fps | 3.274 |
| detect_object_yolo | 12.882 |
| audio_scan | 16.120 |
| asr_timings | 10.144 |
| ast_timings | 55.456 |
| describe_scenes | 24.091 |
| summarize_scenes | 15.701 |
| synthesize_synopsis | 29.266 |
| make_embedding | 6.580 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 74.704 |
| branch_yolo_total | 16.162 |
| branch_audio_total | 81.729 |
