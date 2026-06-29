# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 22:27:49 UTC | DIgv-e18OzA_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 112.262 | 0.694 | 35.204 | 9.385 | 8.919 | 14.556 | 2.987 |

## 2026-06-24 22:27:49 UTC | DIgv-e18OzA_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/DIgv-e18OzA_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `112.262` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.694 |
| save_clips | - |
| sample_frames | 0.692 |
| caption_frames | 34.506 |
| sample_fps | 1.994 |
| detect_object_yolo | 8.216 |
| audio_scan | 3.937 |
| asr_timings | 0.000 |
| ast_timings | 24.892 |
| describe_scenes | 9.385 |
| summarize_scenes | 8.919 |
| synthesize_synopsis | 14.556 |
| make_embedding | 2.987 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.204 |
| branch_yolo_total | 10.216 |
| branch_audio_total | 28.837 |
