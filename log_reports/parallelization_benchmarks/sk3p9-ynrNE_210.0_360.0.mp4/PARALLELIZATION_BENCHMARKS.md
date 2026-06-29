# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 20:52:07 UTC | sk3p9-ynrNE_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 2164.222 | 0.795 | 2060.021 | 17.532 | 10.297 | 12.904 | 3.876 |

## 2026-06-26 20:52:07 UTC | sk3p9-ynrNE_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/sk3p9-ynrNE_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `2164.222` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.795 |
| save_clips | - |
| sample_frames | 1.701 |
| caption_frames | 43.549 |
| sample_fps | 2.420 |
| detect_object_yolo | 9.667 |
| audio_scan | 12.895 |
| asr_timings | 2014.466 |
| ast_timings | 32.651 |
| describe_scenes | 17.532 |
| summarize_scenes | 10.297 |
| synthesize_synopsis | 12.904 |
| make_embedding | 3.876 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.256 |
| branch_yolo_total | 12.093 |
| branch_audio_total | 2060.021 |
