# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 01:17:37 UTC | uZENK8MxITo_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 136.672 | 0.759 | 53.587 | 12.087 | 8.834 | 7.787 | 3.545 |

## 2026-06-27 01:17:37 UTC | uZENK8MxITo_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/uZENK8MxITo_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `136.672` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.759 |
| save_clips | - |
| sample_frames | 1.202 |
| caption_frames | 35.834 |
| sample_fps | 2.322 |
| detect_object_yolo | 9.305 |
| audio_scan | 14.240 |
| asr_timings | 9.322 |
| ast_timings | 30.017 |
| describe_scenes | 12.087 |
| summarize_scenes | 8.834 |
| synthesize_synopsis | 7.787 |
| make_embedding | 3.545 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.042 |
| branch_yolo_total | 11.634 |
| branch_audio_total | 53.587 |
