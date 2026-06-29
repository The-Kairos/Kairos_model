# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 12:32:31 UTC | 0vQvjLp_b4w_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 103.545 | 0.754 | 35.729 | 9.629 | 6.994 | 21.422 | 2.160 |
| 2026-06-27 14:22:06 UTC | 0vQvjLp_b4w_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 85.180 | 0.853 | 36.556 | 4.723 | 5.785 | 7.370 | 2.111 |

## 2026-06-23 12:32:31 UTC | 0vQvjLp_b4w_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0vQvjLp_b4w_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `103.545` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.754 |
| save_clips | - |
| sample_frames | 0.378 |
| caption_frames | 16.785 |
| sample_fps | 1.882 |
| detect_object_yolo | 6.474 |
| audio_scan | 13.679 |
| asr_timings | 9.860 |
| ast_timings | 12.181 |
| describe_scenes | 9.629 |
| summarize_scenes | 6.994 |
| synthesize_synopsis | 21.422 |
| make_embedding | 2.160 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 17.169 |
| branch_yolo_total | 8.361 |
| branch_audio_total | 35.729 |

## 2026-06-27 14:22:06 UTC | 0vQvjLp_b4w_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0vQvjLp_b4w_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `85.180` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.853 |
| save_clips | - |
| sample_frames | 0.385 |
| caption_frames | 17.427 |
| sample_fps | 1.897 |
| detect_object_yolo | 6.679 |
| audio_scan | 13.886 |
| asr_timings | 10.155 |
| ast_timings | 12.507 |
| describe_scenes | 4.723 |
| summarize_scenes | 5.785 |
| synthesize_synopsis | 7.370 |
| make_embedding | 2.111 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 17.818 |
| branch_yolo_total | 8.582 |
| branch_audio_total | 36.556 |
