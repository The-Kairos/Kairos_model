# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 12:18:22 UTC | 6MTL7aBxgR0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 128.784 | 0.791 | 38.711 | 13.954 | 11.741 | 13.917 | 3.590 |

## 2026-06-24 12:18:22 UTC | 6MTL7aBxgR0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/6MTL7aBxgR0_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `128.784` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.791 |
| save_clips | - |
| sample_frames | 1.208 |
| caption_frames | 37.497 |
| sample_fps | 2.266 |
| detect_object_yolo | 8.681 |
| audio_scan | 3.842 |
| asr_timings | 0.000 |
| ast_timings | 29.892 |
| describe_scenes | 13.954 |
| summarize_scenes | 11.741 |
| synthesize_synopsis | 13.917 |
| make_embedding | 3.590 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.711 |
| branch_yolo_total | 10.952 |
| branch_audio_total | 33.742 |
