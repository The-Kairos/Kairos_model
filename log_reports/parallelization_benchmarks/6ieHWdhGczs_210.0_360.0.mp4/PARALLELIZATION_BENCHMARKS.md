# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 12:39:18 UTC | 6ieHWdhGczs_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 209.308 | 0.780 | 104.316 | 16.216 | 9.447 | 26.132 | 3.449 |

## 2026-06-24 12:39:18 UTC | 6ieHWdhGczs_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/6ieHWdhGczs_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `209.308` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.780 |
| save_clips | - |
| sample_frames | 1.623 |
| caption_frames | 35.238 |
| sample_fps | 2.328 |
| detect_object_yolo | 8.395 |
| audio_scan | 11.762 |
| asr_timings | 66.151 |
| ast_timings | 26.394 |
| describe_scenes | 16.216 |
| summarize_scenes | 9.447 |
| synthesize_synopsis | 26.132 |
| make_embedding | 3.449 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.869 |
| branch_yolo_total | 10.729 |
| branch_audio_total | 104.316 |
