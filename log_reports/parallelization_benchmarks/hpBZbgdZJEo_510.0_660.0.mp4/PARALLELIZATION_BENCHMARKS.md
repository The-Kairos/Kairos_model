# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 07:16:32 UTC | hpBZbgdZJEo_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 153.925 | 0.825 | 52.061 | 21.007 | 10.268 | 19.730 | 3.013 |

## 2026-06-26 07:16:32 UTC | hpBZbgdZJEo_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/hpBZbgdZJEo_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `153.925` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.825 |
| save_clips | - |
| sample_frames | 0.954 |
| caption_frames | 34.237 |
| sample_fps | 2.214 |
| detect_object_yolo | 8.193 |
| audio_scan | 13.919 |
| asr_timings | 13.810 |
| ast_timings | 24.324 |
| describe_scenes | 21.007 |
| summarize_scenes | 10.268 |
| synthesize_synopsis | 19.730 |
| make_embedding | 3.013 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.198 |
| branch_yolo_total | 10.413 |
| branch_audio_total | 52.061 |
