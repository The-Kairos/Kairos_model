# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 21:44:48 UTC | XXDq63KUBNo_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 190.176 | 0.806 | 58.161 | 16.181 | 37.106 | 11.201 | 4.216 |

## 2026-06-25 21:44:48 UTC | XXDq63KUBNo_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/XXDq63KUBNo_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `190.176` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.806 |
| save_clips | - |
| sample_frames | 1.386 |
| caption_frames | 47.468 |
| sample_fps | 2.450 |
| detect_object_yolo | 9.793 |
| audio_scan | 15.038 |
| asr_timings | 7.348 |
| ast_timings | 35.767 |
| describe_scenes | 16.181 |
| summarize_scenes | 37.106 |
| synthesize_synopsis | 11.201 |
| make_embedding | 4.216 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.860 |
| branch_yolo_total | 12.249 |
| branch_audio_total | 58.161 |
