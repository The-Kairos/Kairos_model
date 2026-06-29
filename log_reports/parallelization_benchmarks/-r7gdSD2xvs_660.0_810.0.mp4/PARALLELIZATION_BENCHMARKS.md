# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 13:04:45 UTC | -r7gdSD2xvs_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 128.777 | 0.636 | 48.127 | 8.067 | 19.135 | 9.625 | 2.837 |

## 2026-06-27 13:04:45 UTC | -r7gdSD2xvs_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-r7gdSD2xvs_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `128.777` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.636 |
| save_clips | - |
| sample_frames | 0.743 |
| caption_frames | 28.967 |
| sample_fps | 1.939 |
| detect_object_yolo | 7.277 |
| audio_scan | 14.946 |
| asr_timings | 11.760 |
| ast_timings | 21.412 |
| describe_scenes | 8.067 |
| summarize_scenes | 19.135 |
| synthesize_synopsis | 9.625 |
| make_embedding | 2.837 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.716 |
| branch_yolo_total | 9.222 |
| branch_audio_total | 48.127 |
