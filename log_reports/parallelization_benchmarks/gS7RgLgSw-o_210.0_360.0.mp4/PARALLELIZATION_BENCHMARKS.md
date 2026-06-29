# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 05:10:23 UTC | gS7RgLgSw-o_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 167.222 | 0.784 | 88.885 | 9.717 | 6.692 | 10.802 | 3.086 |

## 2026-06-26 05:10:23 UTC | gS7RgLgSw-o_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/gS7RgLgSw-o_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `167.222` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.784 |
| save_clips | - |
| sample_frames | 0.882 |
| caption_frames | 34.122 |
| sample_fps | 2.210 |
| detect_object_yolo | 8.559 |
| audio_scan | 8.814 |
| asr_timings | 55.538 |
| ast_timings | 24.525 |
| describe_scenes | 9.717 |
| summarize_scenes | 6.692 |
| synthesize_synopsis | 10.802 |
| make_embedding | 3.086 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.010 |
| branch_yolo_total | 10.775 |
| branch_audio_total | 88.885 |
