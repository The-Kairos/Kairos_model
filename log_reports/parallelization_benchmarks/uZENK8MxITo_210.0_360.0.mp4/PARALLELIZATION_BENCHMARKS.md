# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 01:15:19 UTC | uZENK8MxITo_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 167.438 | 0.897 | 94.454 | 16.066 | 9.020 | 10.851 | 2.660 |

## 2026-06-27 01:15:19 UTC | uZENK8MxITo_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/uZENK8MxITo_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `167.438` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.897 |
| save_clips | - |
| sample_frames | 0.709 |
| caption_frames | 17.587 |
| sample_fps | 2.060 |
| detect_object_yolo | 7.310 |
| audio_scan | 17.860 |
| asr_timings | 58.335 |
| ast_timings | 18.252 |
| describe_scenes | 16.066 |
| summarize_scenes | 9.020 |
| synthesize_synopsis | 10.851 |
| make_embedding | 2.660 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 18.301 |
| branch_yolo_total | 9.376 |
| branch_audio_total | 94.454 |
