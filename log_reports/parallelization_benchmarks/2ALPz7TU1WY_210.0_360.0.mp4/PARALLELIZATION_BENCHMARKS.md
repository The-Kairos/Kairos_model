# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 14:14:06 UTC | 2ALPz7TU1WY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 192.111 | 0.761 | 54.198 | 37.043 | 14.067 | 25.909 | 3.997 |
| 2026-06-27 15:34:17 UTC | 2ALPz7TU1WY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 139.487 | 0.769 | 54.501 | 10.344 | 6.174 | 6.760 | 3.903 |

## 2026-06-23 14:14:06 UTC | 2ALPz7TU1WY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2ALPz7TU1WY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `192.111` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.761 |
| save_clips | - |
| sample_frames | 0.996 |
| caption_frames | 42.648 |
| sample_fps | 2.275 |
| detect_object_yolo | 8.852 |
| audio_scan | 12.704 |
| asr_timings | 9.874 |
| ast_timings | 31.611 |
| describe_scenes | 37.043 |
| summarize_scenes | 14.067 |
| synthesize_synopsis | 25.909 |
| make_embedding | 3.997 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.651 |
| branch_yolo_total | 11.132 |
| branch_audio_total | 54.198 |

## 2026-06-27 15:34:17 UTC | 2ALPz7TU1WY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2ALPz7TU1WY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `139.487` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.769 |
| save_clips | - |
| sample_frames | 1.006 |
| caption_frames | 43.315 |
| sample_fps | 2.268 |
| detect_object_yolo | 9.043 |
| audio_scan | 12.868 |
| asr_timings | 9.760 |
| ast_timings | 31.865 |
| describe_scenes | 10.344 |
| summarize_scenes | 6.174 |
| synthesize_synopsis | 6.760 |
| make_embedding | 3.903 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.327 |
| branch_yolo_total | 11.317 |
| branch_audio_total | 54.501 |
