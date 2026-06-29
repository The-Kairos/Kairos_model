# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 12:56:30 UTC | -r7gdSD2xvs_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 147.056 | 0.639 | 57.497 | 9.850 | 7.771 | 15.874 | 3.569 |

## 2026-06-27 12:56:30 UTC | -r7gdSD2xvs_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-r7gdSD2xvs_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `147.056` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.639 |
| save_clips | - |
| sample_frames | 0.889 |
| caption_frames | 39.193 |
| sample_fps | 2.004 |
| detect_object_yolo | 8.338 |
| audio_scan | 13.946 |
| asr_timings | 13.657 |
| ast_timings | 29.885 |
| describe_scenes | 9.850 |
| summarize_scenes | 7.771 |
| synthesize_synopsis | 15.874 |
| make_embedding | 3.569 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.088 |
| branch_yolo_total | 10.348 |
| branch_audio_total | 57.497 |
