# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 01:15:09 UTC | bWQ_wOvZ5Qo_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 165.705 | 0.817 | 61.674 | 13.152 | 13.322 | 10.905 | 4.122 |

## 2026-06-26 01:15:09 UTC | bWQ_wOvZ5Qo_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/bWQ_wOvZ5Qo_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `165.705` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.817 |
| save_clips | - |
| sample_frames | 1.727 |
| caption_frames | 46.502 |
| sample_fps | 2.542 |
| detect_object_yolo | 9.493 |
| audio_scan | 15.023 |
| asr_timings | 11.928 |
| ast_timings | 34.715 |
| describe_scenes | 13.152 |
| summarize_scenes | 13.322 |
| synthesize_synopsis | 10.905 |
| make_embedding | 4.122 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.235 |
| branch_yolo_total | 12.041 |
| branch_audio_total | 61.674 |
