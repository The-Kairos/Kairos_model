# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 04:38:23 UTC | xzmEZo_HUpY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 197.419 | 0.804 | 71.125 | 13.248 | 18.665 | 12.931 | 5.300 |

## 2026-06-27 04:38:23 UTC | xzmEZo_HUpY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/xzmEZo_HUpY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `197.419` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.804 |
| save_clips | - |
| sample_frames | 1.666 |
| caption_frames | 57.766 |
| sample_fps | 2.667 |
| detect_object_yolo | 11.792 |
| audio_scan | 14.067 |
| asr_timings | 13.941 |
| ast_timings | 43.109 |
| describe_scenes | 13.248 |
| summarize_scenes | 18.665 |
| synthesize_synopsis | 12.931 |
| make_embedding | 5.300 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 59.439 |
| branch_yolo_total | 14.465 |
| branch_audio_total | 71.125 |
