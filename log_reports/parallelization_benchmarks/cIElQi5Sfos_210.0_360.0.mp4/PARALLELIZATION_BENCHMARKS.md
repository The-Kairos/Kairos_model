# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 02:09:36 UTC | cIElQi5Sfos_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 183.346 | 0.778 | 64.268 | 17.653 | 8.542 | 17.567 | 5.062 |

## 2026-06-26 02:09:36 UTC | cIElQi5Sfos_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/cIElQi5Sfos_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `183.346` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.778 |
| save_clips | - |
| sample_frames | 1.704 |
| caption_frames | 52.871 |
| sample_fps | 2.589 |
| detect_object_yolo | 10.900 |
| audio_scan | 15.143 |
| asr_timings | 8.195 |
| ast_timings | 40.921 |
| describe_scenes | 17.653 |
| summarize_scenes | 8.542 |
| synthesize_synopsis | 17.567 |
| make_embedding | 5.062 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.581 |
| branch_yolo_total | 13.495 |
| branch_audio_total | 64.268 |
