# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 09:33:57 UTC | itUvi4O0CuU_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 238.739 | 0.783 | 87.937 | 24.605 | 18.588 | 31.688 | 5.101 |

## 2026-06-26 09:33:57 UTC | itUvi4O0CuU_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/itUvi4O0CuU_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `238.739` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.783 |
| save_clips | - |
| sample_frames | 1.665 |
| caption_frames | 53.892 |
| sample_fps | 2.433 |
| detect_object_yolo | 10.638 |
| audio_scan | 10.800 |
| asr_timings | 36.549 |
| ast_timings | 40.579 |
| describe_scenes | 24.605 |
| summarize_scenes | 18.588 |
| synthesize_synopsis | 31.688 |
| make_embedding | 5.101 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 55.563 |
| branch_yolo_total | 13.077 |
| branch_audio_total | 87.937 |
