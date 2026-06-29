# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-10 12:09:40 UTC | spain_vlog.mp4 | parallel | gemini | gemini-embedding-001 | 232.859 | 2.430 | 77.035 | 90.263 | 30.342 | 25.270 | 2.406 |

## 2026-04-10 12:09:40 UTC | spain_vlog.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/c154c871-69bd-44ce-a3a0-e0a91f631b26/spain_vlog.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `232.859` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.430 |
| save_clips | - |
| sample_frames | 5.691 |
| caption_frames | 41.501 |
| sample_fps | 13.358 |
| detect_object_yolo | 18.346 |
| audio_scan | 34.365 |
| asr_timings | 18.304 |
| ast_timings | 42.662 |
| describe_scenes | 90.263 |
| summarize_scenes | 30.342 |
| synthesize_synopsis | 25.270 |
| make_embedding | 2.406 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.199 |
| branch_yolo_total | 31.712 |
| branch_audio_total | 77.035 |
