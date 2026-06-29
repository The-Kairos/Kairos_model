# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 05:31:21 UTC | JxYmILDya0A_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 186.015 | 0.826 | 63.665 | 17.095 | 17.614 | 13.771 | 4.794 |

## 2026-06-25 05:31:21 UTC | JxYmILDya0A_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/JxYmILDya0A_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `186.015` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.826 |
| save_clips | - |
| sample_frames | 1.517 |
| caption_frames | 52.401 |
| sample_fps | 2.508 |
| detect_object_yolo | 10.353 |
| audio_scan | 16.239 |
| asr_timings | 10.407 |
| ast_timings | 37.011 |
| describe_scenes | 17.095 |
| summarize_scenes | 17.614 |
| synthesize_synopsis | 13.771 |
| make_embedding | 4.794 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.924 |
| branch_yolo_total | 12.867 |
| branch_audio_total | 63.665 |
