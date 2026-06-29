# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 13:58:37 UTC | kbhUxAvQEWc_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 182.008 | 0.682 | 54.458 | 13.654 | 33.866 | 22.183 | 3.607 |

## 2026-06-26 13:58:37 UTC | kbhUxAvQEWc_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/kbhUxAvQEWc_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `182.008` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.682 |
| save_clips | - |
| sample_frames | 1.207 |
| caption_frames | 39.643 |
| sample_fps | 2.238 |
| detect_object_yolo | 9.055 |
| audio_scan | 14.012 |
| asr_timings | 10.549 |
| ast_timings | 29.889 |
| describe_scenes | 13.654 |
| summarize_scenes | 33.866 |
| synthesize_synopsis | 22.183 |
| make_embedding | 3.607 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.856 |
| branch_yolo_total | 11.298 |
| branch_audio_total | 54.458 |
