# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 06:51:12 UTC | hj9Vh2NagqA_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 237.931 | 0.830 | 69.336 | 20.611 | 48.291 | 26.809 | 4.708 |

## 2026-06-26 06:51:12 UTC | hj9Vh2NagqA_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/hj9Vh2NagqA_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `237.931` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.830 |
| save_clips | - |
| sample_frames | 1.573 |
| caption_frames | 51.665 |
| sample_fps | 2.592 |
| detect_object_yolo | 10.100 |
| audio_scan | 16.213 |
| asr_timings | 14.088 |
| ast_timings | 39.027 |
| describe_scenes | 20.611 |
| summarize_scenes | 48.291 |
| synthesize_synopsis | 26.809 |
| make_embedding | 4.708 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.245 |
| branch_yolo_total | 12.698 |
| branch_audio_total | 69.336 |
