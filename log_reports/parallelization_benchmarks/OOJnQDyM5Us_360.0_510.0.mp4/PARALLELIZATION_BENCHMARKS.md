# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 11:54:21 UTC | OOJnQDyM5Us_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 105.635 | 0.803 | 35.413 | 5.866 | 16.947 | 23.213 | 1.581 |

## 2026-06-25 11:54:21 UTC | OOJnQDyM5Us_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/OOJnQDyM5Us_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `105.635` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.803 |
| save_clips | - |
| sample_frames | 0.356 |
| caption_frames | 11.669 |
| sample_fps | 1.878 |
| detect_object_yolo | 6.495 |
| audio_scan | 12.360 |
| asr_timings | 15.740 |
| ast_timings | 7.304 |
| describe_scenes | 5.866 |
| summarize_scenes | 16.947 |
| synthesize_synopsis | 23.213 |
| make_embedding | 1.581 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 12.031 |
| branch_yolo_total | 8.379 |
| branch_audio_total | 35.413 |
