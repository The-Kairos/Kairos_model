# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 14:16:17 UTC | 7YEQ0iDR4sw_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 174.962 | 0.779 | 56.342 | 13.784 | 20.039 | 25.991 | 3.628 |

## 2026-06-24 14:16:17 UTC | 7YEQ0iDR4sw_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/7YEQ0iDR4sw_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `174.962` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.779 |
| save_clips | - |
| sample_frames | 1.295 |
| caption_frames | 40.727 |
| sample_fps | 2.260 |
| detect_object_yolo | 8.690 |
| audio_scan | 10.731 |
| asr_timings | 16.214 |
| ast_timings | 29.388 |
| describe_scenes | 13.784 |
| summarize_scenes | 20.039 |
| synthesize_synopsis | 25.991 |
| make_embedding | 3.628 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.028 |
| branch_yolo_total | 10.956 |
| branch_audio_total | 56.342 |
