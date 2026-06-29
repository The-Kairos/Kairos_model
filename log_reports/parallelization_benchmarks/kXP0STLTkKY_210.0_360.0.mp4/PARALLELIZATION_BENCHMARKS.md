# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 13:53:34 UTC | kXP0STLTkKY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 160.158 | 0.796 | 49.654 | 18.846 | 21.507 | 19.823 | 3.087 |

## 2026-06-26 13:53:34 UTC | kXP0STLTkKY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/kXP0STLTkKY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `160.158` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.796 |
| save_clips | - |
| sample_frames | 0.769 |
| caption_frames | 34.223 |
| sample_fps | 2.127 |
| detect_object_yolo | 7.916 |
| audio_scan | 12.969 |
| asr_timings | 11.772 |
| ast_timings | 24.905 |
| describe_scenes | 18.846 |
| summarize_scenes | 21.507 |
| synthesize_synopsis | 19.823 |
| make_embedding | 3.087 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.997 |
| branch_yolo_total | 10.049 |
| branch_audio_total | 49.654 |
