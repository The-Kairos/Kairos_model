# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 21:38:57 UTC | XXDq63KUBNo_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 182.740 | 0.828 | 60.765 | 18.159 | 15.535 | 13.832 | 4.972 |

## 2026-06-25 21:38:57 UTC | XXDq63KUBNo_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/XXDq63KUBNo_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `182.740` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.828 |
| save_clips | - |
| sample_frames | 1.585 |
| caption_frames | 52.877 |
| sample_fps | 2.539 |
| detect_object_yolo | 10.231 |
| audio_scan | 11.840 |
| asr_timings | 8.663 |
| ast_timings | 40.253 |
| describe_scenes | 18.159 |
| summarize_scenes | 15.535 |
| synthesize_synopsis | 13.832 |
| make_embedding | 4.972 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.469 |
| branch_yolo_total | 12.775 |
| branch_audio_total | 60.765 |
