# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 08:47:32 UTC | iLoaQNtHdwc_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 143.316 | 0.652 | 48.608 | 17.752 | 9.525 | 18.203 | 3.159 |

## 2026-06-26 08:47:32 UTC | iLoaQNtHdwc_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/iLoaQNtHdwc_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `143.316` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.652 |
| save_clips | - |
| sample_frames | 0.906 |
| caption_frames | 33.025 |
| sample_fps | 2.011 |
| detect_object_yolo | 8.050 |
| audio_scan | 15.046 |
| asr_timings | 9.420 |
| ast_timings | 24.134 |
| describe_scenes | 17.752 |
| summarize_scenes | 9.525 |
| synthesize_synopsis | 18.203 |
| make_embedding | 3.159 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.937 |
| branch_yolo_total | 10.066 |
| branch_audio_total | 48.608 |
