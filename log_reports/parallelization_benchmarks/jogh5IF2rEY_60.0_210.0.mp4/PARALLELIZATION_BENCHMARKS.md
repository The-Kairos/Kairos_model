# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 11:59:10 UTC | jogh5IF2rEY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 161.285 | 0.792 | 51.963 | 17.693 | 18.039 | 25.780 | 3.031 |

## 2026-06-26 11:59:10 UTC | jogh5IF2rEY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jogh5IF2rEY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `161.285` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.792 |
| save_clips | - |
| sample_frames | 0.683 |
| caption_frames | 31.818 |
| sample_fps | 2.092 |
| detect_object_yolo | 7.982 |
| audio_scan | 15.019 |
| asr_timings | 12.905 |
| ast_timings | 24.030 |
| describe_scenes | 17.693 |
| summarize_scenes | 18.039 |
| synthesize_synopsis | 25.780 |
| make_embedding | 3.031 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.508 |
| branch_yolo_total | 10.079 |
| branch_audio_total | 51.963 |
