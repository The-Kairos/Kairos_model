# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 03:03:13 UTC | dcAFJS34zkE_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 167.587 | 0.792 | 59.657 | 10.998 | 14.693 | 14.296 | 4.227 |

## 2026-06-26 03:03:13 UTC | dcAFJS34zkE_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/dcAFJS34zkE_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `167.587` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.792 |
| save_clips | - |
| sample_frames | 1.265 |
| caption_frames | 48.081 |
| sample_fps | 2.455 |
| detect_object_yolo | 9.696 |
| audio_scan | 14.184 |
| asr_timings | 8.702 |
| ast_timings | 36.763 |
| describe_scenes | 10.998 |
| summarize_scenes | 14.693 |
| synthesize_synopsis | 14.296 |
| make_embedding | 4.227 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.352 |
| branch_yolo_total | 12.156 |
| branch_audio_total | 59.657 |
