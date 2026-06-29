# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 07:24:52 UTC | LldGeCXP6RQ_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 167.484 | 0.758 | 51.051 | 16.095 | 26.674 | 24.275 | 3.081 |

## 2026-06-25 07:24:52 UTC | LldGeCXP6RQ_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/LldGeCXP6RQ_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `167.484` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.758 |
| save_clips | - |
| sample_frames | 0.876 |
| caption_frames | 33.074 |
| sample_fps | 2.102 |
| detect_object_yolo | 8.103 |
| audio_scan | 15.910 |
| asr_timings | 12.017 |
| ast_timings | 23.115 |
| describe_scenes | 16.095 |
| summarize_scenes | 26.674 |
| synthesize_synopsis | 24.275 |
| make_embedding | 3.081 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.956 |
| branch_yolo_total | 10.211 |
| branch_audio_total | 51.051 |
