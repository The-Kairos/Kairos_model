# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 01:03:01 UTC | G19tOR4S2fM_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 152.566 | 0.683 | 51.684 | 13.975 | 21.629 | 11.007 | 3.400 |

## 2026-06-25 01:03:01 UTC | G19tOR4S2fM_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/G19tOR4S2fM_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `152.566` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.683 |
| save_clips | - |
| sample_frames | 1.052 |
| caption_frames | 36.746 |
| sample_fps | 2.070 |
| detect_object_yolo | 8.910 |
| audio_scan | 15.005 |
| asr_timings | 9.430 |
| ast_timings | 27.241 |
| describe_scenes | 13.975 |
| summarize_scenes | 21.629 |
| synthesize_synopsis | 11.007 |
| make_embedding | 3.400 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.803 |
| branch_yolo_total | 10.987 |
| branch_audio_total | 51.684 |
