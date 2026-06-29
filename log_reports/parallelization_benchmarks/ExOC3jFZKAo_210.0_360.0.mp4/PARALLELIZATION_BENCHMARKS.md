# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 00:06:34 UTC | ExOC3jFZKAo_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 227.684 | 0.856 | 76.303 | 19.796 | 18.666 | 12.457 | 6.599 |

## 2026-06-25 00:06:34 UTC | ExOC3jFZKAo_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ExOC3jFZKAo_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `227.684` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.856 |
| save_clips | - |
| sample_frames | 2.080 |
| caption_frames | 73.232 |
| sample_fps | 2.933 |
| detect_object_yolo | 13.327 |
| audio_scan | 9.711 |
| asr_timings | 11.415 |
| ast_timings | 55.169 |
| describe_scenes | 19.796 |
| summarize_scenes | 18.666 |
| synthesize_synopsis | 12.457 |
| make_embedding | 6.599 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 75.318 |
| branch_yolo_total | 16.266 |
| branch_audio_total | 76.303 |
