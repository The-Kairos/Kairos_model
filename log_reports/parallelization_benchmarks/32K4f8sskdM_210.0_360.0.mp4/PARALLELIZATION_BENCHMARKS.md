# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 15:25:14 UTC | 32K4f8sskdM_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 225.682 | 0.673 | 90.556 | 26.286 | 18.017 | 29.138 | 4.400 |
| 2026-06-24 09:23:35 UTC | 32K4f8sskdM_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 201.644 | 0.663 | 80.150 | 19.762 | 21.690 | 19.036 | 3.864 |

## 2026-06-23 15:25:14 UTC | 32K4f8sskdM_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/32K4f8sskdM_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `225.682` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.673 |
| save_clips | - |
| sample_frames | 1.527 |
| caption_frames | 42.101 |
| sample_fps | 2.205 |
| detect_object_yolo | 9.376 |
| audio_scan | 15.758 |
| asr_timings | 42.645 |
| ast_timings | 32.145 |
| describe_scenes | 26.286 |
| summarize_scenes | 18.017 |
| synthesize_synopsis | 29.138 |
| make_embedding | 4.400 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.634 |
| branch_yolo_total | 11.587 |
| branch_audio_total | 90.556 |

## 2026-06-24 09:23:35 UTC | 32K4f8sskdM_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/32K4f8sskdM_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `201.644` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.663 |
| save_clips | - |
| sample_frames | 1.510 |
| caption_frames | 41.824 |
| sample_fps | 2.258 |
| detect_object_yolo | 9.470 |
| audio_scan | 15.956 |
| asr_timings | 32.064 |
| ast_timings | 32.121 |
| describe_scenes | 19.762 |
| summarize_scenes | 21.690 |
| synthesize_synopsis | 19.036 |
| make_embedding | 3.864 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.341 |
| branch_yolo_total | 11.735 |
| branch_audio_total | 80.150 |
