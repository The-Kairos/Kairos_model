# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 11:21:11 UTC | O2FF_trMWS4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 195.301 | 0.612 | 54.835 | 27.399 | 21.116 | 23.977 | 4.174 |

## 2026-06-25 11:21:11 UTC | O2FF_trMWS4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/O2FF_trMWS4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `195.301` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.612 |
| save_clips | - |
| sample_frames | 1.135 |
| caption_frames | 49.423 |
| sample_fps | 2.045 |
| detect_object_yolo | 9.150 |
| audio_scan | 11.483 |
| asr_timings | 6.632 |
| ast_timings | 36.712 |
| describe_scenes | 27.399 |
| summarize_scenes | 21.116 |
| synthesize_synopsis | 23.977 |
| make_embedding | 4.174 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.564 |
| branch_yolo_total | 11.201 |
| branch_audio_total | 54.835 |
