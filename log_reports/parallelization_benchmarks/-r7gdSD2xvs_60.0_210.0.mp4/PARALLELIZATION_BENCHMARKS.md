# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 13:02:35 UTC | -r7gdSD2xvs_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 155.158 | 0.626 | 55.164 | 11.893 | 14.359 | 6.312 | 4.195 |

## 2026-06-27 13:02:35 UTC | -r7gdSD2xvs_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-r7gdSD2xvs_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `155.158` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.626 |
| save_clips | - |
| sample_frames | 1.362 |
| caption_frames | 47.749 |
| sample_fps | 2.234 |
| detect_object_yolo | 9.821 |
| audio_scan | 7.544 |
| asr_timings | 11.895 |
| ast_timings | 35.716 |
| describe_scenes | 11.893 |
| summarize_scenes | 14.359 |
| synthesize_synopsis | 6.312 |
| make_embedding | 4.195 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.117 |
| branch_yolo_total | 12.061 |
| branch_audio_total | 55.164 |
