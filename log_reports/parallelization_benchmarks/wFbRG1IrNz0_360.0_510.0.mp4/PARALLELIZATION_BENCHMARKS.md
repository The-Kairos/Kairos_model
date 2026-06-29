# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 02:53:03 UTC | wFbRG1IrNz0_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 142.616 | 0.765 | 59.213 | 7.904 | 14.402 | 12.134 | 3.056 |

## 2026-06-27 02:53:03 UTC | wFbRG1IrNz0_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/wFbRG1IrNz0_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `142.616` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.765 |
| save_clips | - |
| sample_frames | 0.941 |
| caption_frames | 32.998 |
| sample_fps | 2.175 |
| detect_object_yolo | 7.602 |
| audio_scan | 14.098 |
| asr_timings | 20.984 |
| ast_timings | 24.122 |
| describe_scenes | 7.904 |
| summarize_scenes | 14.402 |
| synthesize_synopsis | 12.134 |
| make_embedding | 3.056 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.945 |
| branch_yolo_total | 9.783 |
| branch_audio_total | 59.213 |
