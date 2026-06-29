# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 09:21:29 UTC | 23YBs2JxE-k_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 82.021 | 1.959 | 34.759 | 2.869 | 4.770 | 6.894 | 1.855 |
| 2026-06-21 20:54:13 UTC | 23YBs2JxE-k_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | - |
| 2026-06-22 13:34:47 UTC | 23YBs2JxE-k_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 124.524 | 1.947 | 35.256 | 10.592 | 14.331 | 30.469 | 1.817 |

## 2026-06-21 09:21:29 UTC | 23YBs2JxE-k_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/23YBs2JxE-k_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `82.021` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.959 |
| save_clips | - |
| sample_frames | 1.285 |
| caption_frames | 14.610 |
| sample_fps | 5.490 |
| detect_object_yolo | 6.250 |
| audio_scan | 13.759 |
| asr_timings | 11.391 |
| ast_timings | 9.601 |
| describe_scenes | 2.869 |
| summarize_scenes | 4.770 |
| synthesize_synopsis | 6.894 |
| make_embedding | 1.855 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 15.901 |
| branch_yolo_total | 11.746 |
| branch_audio_total | 34.759 |

## 2026-06-21 20:54:13 UTC | 23YBs2JxE-k_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/23YBs2JxE-k_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `0.060` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | - |
| save_clips | - |
| sample_frames | - |
| caption_frames | - |
| sample_fps | - |
| detect_object_yolo | - |
| audio_scan | - |
| asr_timings | - |
| ast_timings | - |
| describe_scenes | - |
| summarize_scenes | - |
| synthesize_synopsis | - |
| make_embedding | - |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | - |
| branch_yolo_total | - |
| branch_audio_total | - |

## 2026-06-22 13:34:47 UTC | 23YBs2JxE-k_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/23YBs2JxE-k_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `124.524` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.947 |
| save_clips | - |
| sample_frames | 1.289 |
| caption_frames | 15.441 |
| sample_fps | 5.573 |
| detect_object_yolo | 6.418 |
| audio_scan | 13.956 |
| asr_timings | 11.507 |
| ast_timings | 9.784 |
| describe_scenes | 10.592 |
| summarize_scenes | 14.331 |
| synthesize_synopsis | 30.469 |
| make_embedding | 1.817 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 16.736 |
| branch_yolo_total | 11.996 |
| branch_audio_total | 35.256 |
