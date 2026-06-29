# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 03:06:18 UTC | H7jdqCKg9x8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 1963.506 | 0.775 | 1904.284 | 7.937 | 6.115 | 9.717 | 2.275 |

## 2026-06-25 03:06:18 UTC | H7jdqCKg9x8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/H7jdqCKg9x8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `1963.506` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.775 |
| save_clips | - |
| sample_frames | 0.626 |
| caption_frames | 21.530 |
| sample_fps | 2.020 |
| detect_object_yolo | 6.842 |
| audio_scan | 12.688 |
| asr_timings | 1876.073 |
| ast_timings | 15.513 |
| describe_scenes | 7.937 |
| summarize_scenes | 6.115 |
| synthesize_synopsis | 9.717 |
| make_embedding | 2.275 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 22.163 |
| branch_yolo_total | 8.867 |
| branch_audio_total | 1904.284 |
