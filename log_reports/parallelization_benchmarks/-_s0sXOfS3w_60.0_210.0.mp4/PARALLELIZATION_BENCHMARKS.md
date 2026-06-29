# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 20:53:30 UTC | -_s0sXOfS3w_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | - |
| 2026-06-22 11:41:47 UTC | -_s0sXOfS3w_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 140.367 | 0.767 | 58.793 | 9.823 | 6.085 | 24.642 | 2.580 |

## 2026-06-21 20:53:30 UTC | -_s0sXOfS3w_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-_s0sXOfS3w_60.0_210.0.mp4`
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

## 2026-06-22 11:41:47 UTC | -_s0sXOfS3w_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-_s0sXOfS3w_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `140.367` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.767 |
| save_clips | - |
| sample_frames | 0.661 |
| caption_frames | 25.965 |
| sample_fps | 2.011 |
| detect_object_yolo | 7.666 |
| audio_scan | 15.946 |
| asr_timings | 24.251 |
| ast_timings | 18.587 |
| describe_scenes | 9.823 |
| summarize_scenes | 6.085 |
| synthesize_synopsis | 24.642 |
| make_embedding | 2.580 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.632 |
| branch_yolo_total | 9.683 |
| branch_audio_total | 58.793 |
