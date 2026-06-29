# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 12:30:46 UTC | 0vQvjLp_b4w_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 128.247 | 0.772 | 40.917 | 12.770 | 10.855 | 27.012 | 2.348 |
| 2026-06-27 14:20:39 UTC | 0vQvjLp_b4w_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 103.279 | 0.803 | 41.397 | 5.492 | 12.183 | 7.810 | 2.274 |

## 2026-06-23 12:30:46 UTC | 0vQvjLp_b4w_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0vQvjLp_b4w_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `128.247` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.772 |
| save_clips | - |
| sample_frames | 0.571 |
| caption_frames | 22.483 |
| sample_fps | 1.991 |
| detect_object_yolo | 7.163 |
| audio_scan | 14.758 |
| asr_timings | 10.738 |
| ast_timings | 15.413 |
| describe_scenes | 12.770 |
| summarize_scenes | 10.855 |
| synthesize_synopsis | 27.012 |
| make_embedding | 2.348 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.060 |
| branch_yolo_total | 9.160 |
| branch_audio_total | 40.917 |

## 2026-06-27 14:20:39 UTC | 0vQvjLp_b4w_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0vQvjLp_b4w_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `103.279` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.803 |
| save_clips | - |
| sample_frames | 0.584 |
| caption_frames | 22.067 |
| sample_fps | 1.973 |
| detect_object_yolo | 7.295 |
| audio_scan | 14.900 |
| asr_timings | 10.835 |
| ast_timings | 15.652 |
| describe_scenes | 5.492 |
| summarize_scenes | 12.183 |
| synthesize_synopsis | 7.810 |
| make_embedding | 2.274 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 22.657 |
| branch_yolo_total | 9.274 |
| branch_audio_total | 41.397 |
