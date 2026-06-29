# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 15:10:14 UTC | lLepYM0nQhc_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 176.965 | 0.832 | 73.223 | 17.486 | 8.225 | 27.649 | 3.061 |

## 2026-06-26 15:10:14 UTC | lLepYM0nQhc_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/lLepYM0nQhc_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `176.965` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.832 |
| save_clips | - |
| sample_frames | 0.830 |
| caption_frames | 33.895 |
| sample_fps | 2.142 |
| detect_object_yolo | 8.183 |
| audio_scan | 16.018 |
| asr_timings | 32.220 |
| ast_timings | 24.976 |
| describe_scenes | 17.486 |
| summarize_scenes | 8.225 |
| synthesize_synopsis | 27.649 |
| make_embedding | 3.061 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.731 |
| branch_yolo_total | 10.331 |
| branch_audio_total | 73.223 |
