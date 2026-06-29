# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 17:04:17 UTC | SPpXtLSyyDw_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 186.444 | 0.779 | 58.180 | 19.746 | 25.365 | 19.686 | 4.156 |

## 2026-06-25 17:04:17 UTC | SPpXtLSyyDw_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/SPpXtLSyyDw_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `186.444` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.779 |
| save_clips | - |
| sample_frames | 1.231 |
| caption_frames | 43.992 |
| sample_fps | 2.285 |
| detect_object_yolo | 9.619 |
| audio_scan | 12.796 |
| asr_timings | 9.715 |
| ast_timings | 35.660 |
| describe_scenes | 19.746 |
| summarize_scenes | 25.365 |
| synthesize_synopsis | 19.686 |
| make_embedding | 4.156 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.229 |
| branch_yolo_total | 11.910 |
| branch_audio_total | 58.180 |
