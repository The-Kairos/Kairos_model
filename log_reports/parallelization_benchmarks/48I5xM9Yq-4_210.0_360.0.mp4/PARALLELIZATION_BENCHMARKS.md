# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 16:43:10 UTC | 48I5xM9Yq-4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 125.019 | 0.619 | 41.442 | 12.618 | 10.053 | 16.695 | 2.930 |
| 2026-06-24 10:38:18 UTC | 48I5xM9Yq-4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 123.394 | 0.698 | 42.335 | 8.436 | 11.320 | 17.095 | 2.815 |

## 2026-06-23 16:43:10 UTC | 48I5xM9Yq-4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/48I5xM9Yq-4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `125.019` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.619 |
| save_clips | - |
| sample_frames | 0.830 |
| caption_frames | 29.422 |
| sample_fps | 1.907 |
| detect_object_yolo | 7.114 |
| audio_scan | 12.648 |
| asr_timings | 7.186 |
| ast_timings | 21.599 |
| describe_scenes | 12.618 |
| summarize_scenes | 10.053 |
| synthesize_synopsis | 16.695 |
| make_embedding | 2.930 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.258 |
| branch_yolo_total | 9.027 |
| branch_audio_total | 41.442 |

## 2026-06-24 10:38:18 UTC | 48I5xM9Yq-4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/48I5xM9Yq-4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `123.394` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.698 |
| save_clips | - |
| sample_frames | 0.853 |
| caption_frames | 29.335 |
| sample_fps | 1.936 |
| detect_object_yolo | 7.166 |
| audio_scan | 12.752 |
| asr_timings | 8.028 |
| ast_timings | 21.545 |
| describe_scenes | 8.436 |
| summarize_scenes | 11.320 |
| synthesize_synopsis | 17.095 |
| make_embedding | 2.815 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.194 |
| branch_yolo_total | 9.109 |
| branch_audio_total | 42.335 |
