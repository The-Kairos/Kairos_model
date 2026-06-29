# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 20:22:15 UTC | W-LP4gzGiVI_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 98.831 | 0.809 | 36.682 | 4.902 | 6.248 | 18.593 | 2.065 |

## 2026-06-25 20:22:15 UTC | W-LP4gzGiVI_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/W-LP4gzGiVI_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `98.831` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.809 |
| save_clips | - |
| sample_frames | 0.540 |
| caption_frames | 18.919 |
| sample_fps | 1.908 |
| detect_object_yolo | 6.739 |
| audio_scan | 10.696 |
| asr_timings | 12.832 |
| ast_timings | 13.145 |
| describe_scenes | 4.902 |
| summarize_scenes | 6.248 |
| synthesize_synopsis | 18.593 |
| make_embedding | 2.065 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 19.465 |
| branch_yolo_total | 8.653 |
| branch_audio_total | 36.682 |
