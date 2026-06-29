# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 07:00:11 UTC | KyrcDEJmR5w_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 130.082 | 0.896 | 41.231 | 13.505 | 18.174 | 20.289 | 2.350 |

## 2026-06-25 07:00:11 UTC | KyrcDEJmR5w_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/KyrcDEJmR5w_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `130.082` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.896 |
| save_clips | - |
| sample_frames | 0.659 |
| caption_frames | 22.382 |
| sample_fps | 2.021 |
| detect_object_yolo | 7.181 |
| audio_scan | 15.993 |
| asr_timings | 9.831 |
| ast_timings | 15.399 |
| describe_scenes | 13.505 |
| summarize_scenes | 18.174 |
| synthesize_synopsis | 20.289 |
| make_embedding | 2.350 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.047 |
| branch_yolo_total | 9.208 |
| branch_audio_total | 41.231 |
