# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 18:25:14 UTC | r62uMwPmBK0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 110.983 | 0.709 | 43.631 | 23.624 | 14.850 | 10.235 | 3.762 |

## 2026-06-26 18:25:14 UTC | r62uMwPmBK0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/r62uMwPmBK0_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `110.983` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.709 |
| save_clips | - |
| sample_frames | 1.563 |
| caption_frames | 42.063 |
| sample_fps | 2.301 |
| detect_object_yolo | 9.422 |
| audio_scan | 1.042 |
| asr_timings | 0.000 |
| ast_timings | 0.000 |
| describe_scenes | 23.624 |
| summarize_scenes | 14.850 |
| synthesize_synopsis | 10.235 |
| make_embedding | 3.762 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.631 |
| branch_yolo_total | 11.728 |
| branch_audio_total | 1.049 |
