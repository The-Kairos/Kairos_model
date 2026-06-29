# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 02:24:34 UTC | cKKxp83EQp4_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 55.672 | 0.788 | 12.507 | 5.380 | 3.832 | 11.405 | 1.536 |

## 2026-06-26 02:24:34 UTC | cKKxp83EQp4_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/cKKxp83EQp4_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `55.672` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.788 |
| save_clips | - |
| sample_frames | 0.245 |
| caption_frames | 12.256 |
| sample_fps | 1.818 |
| detect_object_yolo | 5.815 |
| audio_scan | 3.867 |
| asr_timings | 0.000 |
| ast_timings | 7.326 |
| describe_scenes | 5.380 |
| summarize_scenes | 3.832 |
| synthesize_synopsis | 11.405 |
| make_embedding | 1.536 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 12.507 |
| branch_yolo_total | 7.639 |
| branch_audio_total | 11.201 |
