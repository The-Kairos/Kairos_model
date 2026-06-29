# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 22:49:19 UTC | Zhx9Ki9bUkE_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 147.766 | 0.787 | 57.164 | 10.539 | 10.316 | 12.482 | 3.565 |

## 2026-06-25 22:49:19 UTC | Zhx9Ki9bUkE_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Zhx9Ki9bUkE_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `147.766` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.787 |
| save_clips | - |
| sample_frames | 1.010 |
| caption_frames | 39.520 |
| sample_fps | 2.251 |
| detect_object_yolo | 8.732 |
| audio_scan | 14.786 |
| asr_timings | 12.190 |
| ast_timings | 30.180 |
| describe_scenes | 10.539 |
| summarize_scenes | 10.316 |
| synthesize_synopsis | 12.482 |
| make_embedding | 3.565 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.536 |
| branch_yolo_total | 10.989 |
| branch_audio_total | 57.164 |
