# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 01:22:50 UTC | bXBDZfrEKzk_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 134.025 | 0.798 | 52.148 | 11.605 | 8.415 | 10.153 | 3.065 |

## 2026-06-26 01:22:50 UTC | bXBDZfrEKzk_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/bXBDZfrEKzk_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `134.025` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.798 |
| save_clips | - |
| sample_frames | 1.069 |
| caption_frames | 35.439 |
| sample_fps | 2.218 |
| detect_object_yolo | 7.720 |
| audio_scan | 13.850 |
| asr_timings | 13.745 |
| ast_timings | 24.545 |
| describe_scenes | 11.605 |
| summarize_scenes | 8.415 |
| synthesize_synopsis | 10.153 |
| make_embedding | 3.065 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.514 |
| branch_yolo_total | 9.944 |
| branch_audio_total | 52.148 |
