# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 17:27:59 UTC | 4yY-cFoIYfM_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 132.019 | 0.740 | 37.986 | 15.924 | 19.074 | 15.940 | 4.699 |
| 2026-06-24 11:20:33 UTC | 4yY-cFoIYfM_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 126.152 | 0.766 | 38.579 | 19.937 | 7.574 | 19.510 | 2.550 |

## 2026-06-23 17:27:59 UTC | 4yY-cFoIYfM_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4yY-cFoIYfM_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `132.019` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.740 |
| save_clips | - |
| sample_frames | 0.520 |
| caption_frames | 27.423 |
| sample_fps | 1.946 |
| detect_object_yolo | 6.396 |
| audio_scan | 11.575 |
| asr_timings | 8.140 |
| ast_timings | 18.263 |
| describe_scenes | 15.924 |
| summarize_scenes | 19.074 |
| synthesize_synopsis | 15.940 |
| make_embedding | 4.699 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.949 |
| branch_yolo_total | 8.347 |
| branch_audio_total | 37.986 |

## 2026-06-24 11:20:33 UTC | 4yY-cFoIYfM_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4yY-cFoIYfM_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `126.152` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.766 |
| save_clips | - |
| sample_frames | 0.513 |
| caption_frames | 26.887 |
| sample_fps | 1.960 |
| detect_object_yolo | 6.477 |
| audio_scan | 11.734 |
| asr_timings | 8.593 |
| ast_timings | 18.243 |
| describe_scenes | 19.937 |
| summarize_scenes | 7.574 |
| synthesize_synopsis | 19.510 |
| make_embedding | 2.550 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.406 |
| branch_yolo_total | 8.443 |
| branch_audio_total | 38.579 |
