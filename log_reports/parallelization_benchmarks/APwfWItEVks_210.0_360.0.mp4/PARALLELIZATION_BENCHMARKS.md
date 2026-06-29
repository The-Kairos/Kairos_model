# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 18:47:50 UTC | APwfWItEVks_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 170.564 | 0.772 | 52.308 | 11.834 | 35.417 | 19.359 | 3.271 |

## 2026-06-24 18:47:50 UTC | APwfWItEVks_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/APwfWItEVks_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `170.564` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.772 |
| save_clips | - |
| sample_frames | 0.828 |
| caption_frames | 34.615 |
| sample_fps | 2.114 |
| detect_object_yolo | 8.637 |
| audio_scan | 14.927 |
| asr_timings | 10.776 |
| ast_timings | 26.596 |
| describe_scenes | 11.834 |
| summarize_scenes | 35.417 |
| synthesize_synopsis | 19.359 |
| make_embedding | 3.271 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.448 |
| branch_yolo_total | 10.758 |
| branch_audio_total | 52.308 |
