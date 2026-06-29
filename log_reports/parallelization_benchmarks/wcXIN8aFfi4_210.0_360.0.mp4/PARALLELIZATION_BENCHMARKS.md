# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 03:13:16 UTC | wcXIN8aFfi4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 156.952 | 0.667 | 54.174 | 12.839 | 13.404 | 10.859 | 4.106 |

## 2026-06-27 03:13:16 UTC | wcXIN8aFfi4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/wcXIN8aFfi4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `156.952` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.667 |
| save_clips | - |
| sample_frames | 1.007 |
| caption_frames | 46.645 |
| sample_fps | 2.138 |
| detect_object_yolo | 9.717 |
| audio_scan | 8.700 |
| asr_timings | 10.047 |
| ast_timings | 35.419 |
| describe_scenes | 12.839 |
| summarize_scenes | 13.404 |
| synthesize_synopsis | 10.859 |
| make_embedding | 4.106 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.658 |
| branch_yolo_total | 11.860 |
| branch_audio_total | 54.174 |
