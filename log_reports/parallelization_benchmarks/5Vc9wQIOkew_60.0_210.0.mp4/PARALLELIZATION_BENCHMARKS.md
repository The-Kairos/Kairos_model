# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 12:00:58 UTC | 5Vc9wQIOkew_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 216.205 | 0.868 | 62.459 | 19.189 | 39.078 | 21.458 | 5.021 |

## 2026-06-24 12:00:58 UTC | 5Vc9wQIOkew_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/5Vc9wQIOkew_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `216.205` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.868 |
| save_clips | - |
| sample_frames | 1.518 |
| caption_frames | 52.315 |
| sample_fps | 2.462 |
| detect_object_yolo | 10.444 |
| audio_scan | 14.899 |
| asr_timings | 7.249 |
| ast_timings | 40.303 |
| describe_scenes | 19.189 |
| summarize_scenes | 39.078 |
| synthesize_synopsis | 21.458 |
| make_embedding | 5.021 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.839 |
| branch_yolo_total | 12.911 |
| branch_audio_total | 62.459 |
