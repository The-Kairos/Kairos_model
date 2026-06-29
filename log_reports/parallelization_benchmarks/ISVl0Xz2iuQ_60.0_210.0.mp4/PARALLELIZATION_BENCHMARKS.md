# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 04:55:18 UTC | ISVl0Xz2iuQ_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 238.162 | 0.686 | 66.641 | 24.521 | 54.234 | 14.081 | 5.414 |

## 2026-06-25 04:55:18 UTC | ISVl0Xz2iuQ_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ISVl0Xz2iuQ_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `238.162` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.686 |
| save_clips | - |
| sample_frames | 1.548 |
| caption_frames | 56.199 |
| sample_fps | 2.459 |
| detect_object_yolo | 10.964 |
| audio_scan | 13.871 |
| asr_timings | 8.987 |
| ast_timings | 43.775 |
| describe_scenes | 24.521 |
| summarize_scenes | 54.234 |
| synthesize_synopsis | 14.081 |
| make_embedding | 5.414 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 57.753 |
| branch_yolo_total | 13.429 |
| branch_audio_total | 66.641 |
