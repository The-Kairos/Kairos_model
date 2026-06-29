# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 02:28:12 UTC | cKKxp83EQp4_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 61.727 | 0.770 | 15.323 | 5.444 | 3.643 | 11.311 | 1.806 |

## 2026-06-26 02:28:12 UTC | cKKxp83EQp4_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/cKKxp83EQp4_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `61.727` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.770 |
| save_clips | - |
| sample_frames | 0.381 |
| caption_frames | 14.936 |
| sample_fps | 1.873 |
| detect_object_yolo | 6.005 |
| audio_scan | 3.894 |
| asr_timings | 0.000 |
| ast_timings | 10.253 |
| describe_scenes | 5.444 |
| summarize_scenes | 3.643 |
| synthesize_synopsis | 11.311 |
| make_embedding | 1.806 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 15.323 |
| branch_yolo_total | 7.884 |
| branch_audio_total | 14.155 |
