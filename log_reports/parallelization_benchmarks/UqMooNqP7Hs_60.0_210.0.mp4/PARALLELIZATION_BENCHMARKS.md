# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 19:27:47 UTC | UqMooNqP7Hs_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 150.715 | 0.791 | 54.520 | 15.309 | 15.729 | 8.510 | 3.645 |

## 2026-06-25 19:27:47 UTC | UqMooNqP7Hs_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/UqMooNqP7Hs_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `150.715` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.791 |
| save_clips | - |
| sample_frames | 1.058 |
| caption_frames | 38.719 |
| sample_fps | 2.193 |
| detect_object_yolo | 8.844 |
| audio_scan | 14.984 |
| asr_timings | 9.685 |
| ast_timings | 29.842 |
| describe_scenes | 15.309 |
| summarize_scenes | 15.729 |
| synthesize_synopsis | 8.510 |
| make_embedding | 3.645 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.783 |
| branch_yolo_total | 11.042 |
| branch_audio_total | 54.520 |
