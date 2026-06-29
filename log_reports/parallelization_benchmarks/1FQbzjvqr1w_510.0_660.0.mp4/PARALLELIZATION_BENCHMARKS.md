# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 13:06:14 UTC | 1FQbzjvqr1w_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 184.449 | 0.805 | 75.790 | 19.998 | 11.724 | 33.507 | 2.850 |
| 2026-06-27 14:46:56 UTC | 1FQbzjvqr1w_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 128.882 | 0.830 | 62.675 | 6.777 | 5.577 | 9.743 | 2.823 |

## 2026-06-23 13:06:14 UTC | 1FQbzjvqr1w_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1FQbzjvqr1w_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `184.449` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.805 |
| save_clips | - |
| sample_frames | 0.955 |
| caption_frames | 28.191 |
| sample_fps | 2.118 |
| detect_object_yolo | 7.124 |
| audio_scan | 14.628 |
| asr_timings | 40.850 |
| ast_timings | 20.303 |
| describe_scenes | 19.998 |
| summarize_scenes | 11.724 |
| synthesize_synopsis | 33.507 |
| make_embedding | 2.850 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.151 |
| branch_yolo_total | 9.247 |
| branch_audio_total | 75.790 |

## 2026-06-27 14:46:56 UTC | 1FQbzjvqr1w_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1FQbzjvqr1w_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `128.882` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.830 |
| save_clips | - |
| sample_frames | 0.974 |
| caption_frames | 28.597 |
| sample_fps | 2.143 |
| detect_object_yolo | 7.306 |
| audio_scan | 15.019 |
| asr_timings | 27.086 |
| ast_timings | 20.561 |
| describe_scenes | 6.777 |
| summarize_scenes | 5.577 |
| synthesize_synopsis | 9.743 |
| make_embedding | 2.823 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.577 |
| branch_yolo_total | 9.455 |
| branch_audio_total | 62.675 |
