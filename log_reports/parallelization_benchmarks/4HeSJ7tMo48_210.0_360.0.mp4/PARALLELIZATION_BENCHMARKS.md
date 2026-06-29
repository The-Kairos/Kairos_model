# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 22:45:06 UTC | 4HeSJ7tMo48_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 182.574 | 0.709 | 91.490 | 10.720 | 6.770 | 6.614 | 4.234 |

## 2026-06-21 22:45:06 UTC | 4HeSJ7tMo48_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4HeSJ7tMo48_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `182.574` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.709 |
| save_clips | - |
| sample_frames | 1.484 |
| caption_frames | 46.961 |
| sample_fps | 2.416 |
| detect_object_yolo | 9.769 |
| audio_scan | 12.886 |
| asr_timings | 43.177 |
| ast_timings | 35.419 |
| describe_scenes | 10.720 |
| summarize_scenes | 6.770 |
| synthesize_synopsis | 6.614 |
| make_embedding | 4.234 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.451 |
| branch_yolo_total | 12.191 |
| branch_audio_total | 91.490 |
