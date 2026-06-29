# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 08:31:24 UTC | iJRwe2oCX0E_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 115.307 | 0.807 | 40.212 | 7.897 | 7.398 | 24.203 | 2.342 |

## 2026-06-26 08:31:24 UTC | iJRwe2oCX0E_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/iJRwe2oCX0E_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `115.307` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.807 |
| save_clips | - |
| sample_frames | 0.576 |
| caption_frames | 21.577 |
| sample_fps | 1.995 |
| detect_object_yolo | 6.911 |
| audio_scan | 12.911 |
| asr_timings | 11.366 |
| ast_timings | 15.926 |
| describe_scenes | 7.897 |
| summarize_scenes | 7.398 |
| synthesize_synopsis | 24.203 |
| make_embedding | 2.342 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 22.159 |
| branch_yolo_total | 8.912 |
| branch_audio_total | 40.212 |
