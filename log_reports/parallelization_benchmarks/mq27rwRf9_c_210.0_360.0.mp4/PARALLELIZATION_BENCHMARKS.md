# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 16:18:55 UTC | mq27rwRf9_c_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 135.558 | 0.661 | 51.126 | 9.683 | 13.679 | 8.399 | 3.364 |

## 2026-06-27 16:18:55 UTC | mq27rwRf9_c_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/mq27rwRf9_c_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `135.558` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.661 |
| save_clips | - |
| sample_frames | 0.873 |
| caption_frames | 36.162 |
| sample_fps | 2.024 |
| detect_object_yolo | 8.195 |
| audio_scan | 15.917 |
| asr_timings | 7.952 |
| ast_timings | 27.248 |
| describe_scenes | 9.683 |
| summarize_scenes | 13.679 |
| synthesize_synopsis | 8.399 |
| make_embedding | 3.364 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.041 |
| branch_yolo_total | 10.224 |
| branch_audio_total | 51.126 |
