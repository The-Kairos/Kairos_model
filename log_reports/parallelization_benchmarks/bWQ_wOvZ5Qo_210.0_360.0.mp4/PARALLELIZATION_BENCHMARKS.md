# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 01:09:45 UTC | bWQ_wOvZ5Qo_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 116.961 | 0.841 | 50.271 | 7.349 | 9.016 | 9.587 | 2.540 |

## 2026-06-26 01:09:45 UTC | bWQ_wOvZ5Qo_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/bWQ_wOvZ5Qo_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `116.961` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.841 |
| save_clips | - |
| sample_frames | 0.735 |
| caption_frames | 26.266 |
| sample_fps | 2.075 |
| detect_object_yolo | 6.797 |
| audio_scan | 15.041 |
| asr_timings | 16.757 |
| ast_timings | 18.464 |
| describe_scenes | 7.349 |
| summarize_scenes | 9.016 |
| synthesize_synopsis | 9.587 |
| make_embedding | 2.540 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.007 |
| branch_yolo_total | 8.878 |
| branch_audio_total | 50.271 |
