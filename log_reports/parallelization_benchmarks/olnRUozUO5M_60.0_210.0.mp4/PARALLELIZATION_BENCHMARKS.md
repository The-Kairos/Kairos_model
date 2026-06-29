# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 07:35:38 UTC | olnRUozUO5M_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 202.761 | 0.792 | 73.310 | 16.011 | 9.319 | 7.461 | 6.681 |

## 2026-06-28 07:35:38 UTC | olnRUozUO5M_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/olnRUozUO5M_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `202.761` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.792 |
| save_clips | - |
| sample_frames | 1.901 |
| caption_frames | 69.680 |
| sample_fps | 2.835 |
| detect_object_yolo | 13.302 |
| audio_scan | 12.936 |
| asr_timings | 9.412 |
| ast_timings | 50.954 |
| describe_scenes | 16.011 |
| summarize_scenes | 9.319 |
| synthesize_synopsis | 7.461 |
| make_embedding | 6.681 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 71.587 |
| branch_yolo_total | 16.143 |
| branch_audio_total | 73.310 |
