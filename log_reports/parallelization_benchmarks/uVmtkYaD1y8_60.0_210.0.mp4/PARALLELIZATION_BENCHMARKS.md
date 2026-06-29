# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 01:01:53 UTC | uVmtkYaD1y8_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 192.191 | 0.631 | 71.696 | 15.095 | 7.596 | 16.967 | 5.377 |

## 2026-06-27 01:01:53 UTC | uVmtkYaD1y8_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/uVmtkYaD1y8_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `192.191` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.631 |
| save_clips | - |
| sample_frames | 1.481 |
| caption_frames | 58.359 |
| sample_fps | 2.331 |
| detect_object_yolo | 11.224 |
| audio_scan | 16.010 |
| asr_timings | 10.694 |
| ast_timings | 44.983 |
| describe_scenes | 15.095 |
| summarize_scenes | 7.596 |
| synthesize_synopsis | 16.967 |
| make_embedding | 5.377 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 59.847 |
| branch_yolo_total | 13.562 |
| branch_audio_total | 71.696 |
