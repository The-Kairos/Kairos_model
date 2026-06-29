# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 01:17:50 UTC | bWQ_wOvZ5Qo_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 160.033 | 0.805 | 59.398 | 13.189 | 13.585 | 11.715 | 3.815 |

## 2026-06-26 01:17:50 UTC | bWQ_wOvZ5Qo_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/bWQ_wOvZ5Qo_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `160.033` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.805 |
| save_clips | - |
| sample_frames | 1.293 |
| caption_frames | 43.239 |
| sample_fps | 2.333 |
| detect_object_yolo | 9.238 |
| audio_scan | 13.833 |
| asr_timings | 12.712 |
| ast_timings | 32.844 |
| describe_scenes | 13.189 |
| summarize_scenes | 13.585 |
| synthesize_synopsis | 11.715 |
| make_embedding | 3.815 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.538 |
| branch_yolo_total | 11.576 |
| branch_audio_total | 59.398 |
