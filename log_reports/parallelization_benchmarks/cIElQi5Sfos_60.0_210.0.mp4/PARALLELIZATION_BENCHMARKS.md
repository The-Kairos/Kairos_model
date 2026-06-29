# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 02:18:08 UTC | cIElQi5Sfos_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 198.100 | 0.793 | 72.784 | 14.952 | 20.261 | 13.721 | 5.108 |

## 2026-06-26 02:18:08 UTC | cIElQi5Sfos_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/cIElQi5Sfos_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `198.100` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.793 |
| save_clips | - |
| sample_frames | 1.448 |
| caption_frames | 54.395 |
| sample_fps | 2.485 |
| detect_object_yolo | 10.728 |
| audio_scan | 16.248 |
| asr_timings | 14.644 |
| ast_timings | 41.883 |
| describe_scenes | 14.952 |
| summarize_scenes | 20.261 |
| synthesize_synopsis | 13.721 |
| make_embedding | 5.108 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 55.849 |
| branch_yolo_total | 13.218 |
| branch_audio_total | 72.784 |
