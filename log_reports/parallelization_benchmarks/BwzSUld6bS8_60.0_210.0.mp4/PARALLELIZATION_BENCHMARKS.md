# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 20:06:15 UTC | BwzSUld6bS8_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 141.112 | 0.674 | 49.548 | 10.828 | 8.869 | 22.686 | 3.066 |

## 2026-06-24 20:06:15 UTC | BwzSUld6bS8_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/BwzSUld6bS8_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `141.112` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.674 |
| save_clips | - |
| sample_frames | 0.931 |
| caption_frames | 33.086 |
| sample_fps | 2.066 |
| detect_object_yolo | 7.941 |
| audio_scan | 15.019 |
| asr_timings | 10.749 |
| ast_timings | 23.772 |
| describe_scenes | 10.828 |
| summarize_scenes | 8.869 |
| synthesize_synopsis | 22.686 |
| make_embedding | 3.066 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.023 |
| branch_yolo_total | 10.013 |
| branch_audio_total | 49.548 |
