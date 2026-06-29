# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 10:22:46 UTC | 3PTyasnWk68_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 161.779 | 1.348 | 60.680 | 10.567 | 8.701 | 7.301 | 4.689 |
| 2026-06-21 22:02:18 UTC | 3PTyasnWk68_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 170.194 | 1.363 | 61.989 | 10.952 | 10.709 | 9.136 | 5.005 |

## 2026-06-21 10:22:46 UTC | 3PTyasnWk68_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3PTyasnWk68_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `161.779` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.348 |
| save_clips | - |
| sample_frames | 3.134 |
| caption_frames | 48.529 |
| sample_fps | 5.718 |
| detect_object_yolo | 9.802 |
| audio_scan | 12.653 |
| asr_timings | 9.734 |
| ast_timings | 38.284 |
| describe_scenes | 10.567 |
| summarize_scenes | 8.701 |
| synthesize_synopsis | 7.301 |
| make_embedding | 4.689 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.668 |
| branch_yolo_total | 15.525 |
| branch_audio_total | 60.680 |

## 2026-06-21 22:02:18 UTC | 3PTyasnWk68_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3PTyasnWk68_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `170.194` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.363 |
| save_clips | - |
| sample_frames | 3.126 |
| caption_frames | 50.076 |
| sample_fps | 5.846 |
| detect_object_yolo | 10.538 |
| audio_scan | 13.048 |
| asr_timings | 9.939 |
| ast_timings | 38.994 |
| describe_scenes | 10.952 |
| summarize_scenes | 10.709 |
| synthesize_synopsis | 9.136 |
| make_embedding | 5.005 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.208 |
| branch_yolo_total | 16.390 |
| branch_audio_total | 61.989 |
