# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 12:11:40 UTC | OdKI-fHG_ZY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 287.482 | 0.682 | 75.949 | 54.063 | 21.881 | 35.975 | 6.691 |

## 2026-06-25 12:11:40 UTC | OdKI-fHG_ZY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/OdKI-fHG_ZY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `287.482` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.682 |
| save_clips | - |
| sample_frames | 1.720 |
| caption_frames | 74.223 |
| sample_fps | 2.545 |
| detect_object_yolo | 13.067 |
| audio_scan | 8.870 |
| asr_timings | 9.712 |
| ast_timings | 56.625 |
| describe_scenes | 54.063 |
| summarize_scenes | 21.881 |
| synthesize_synopsis | 35.975 |
| make_embedding | 6.691 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 75.949 |
| branch_yolo_total | 15.618 |
| branch_audio_total | 75.216 |
