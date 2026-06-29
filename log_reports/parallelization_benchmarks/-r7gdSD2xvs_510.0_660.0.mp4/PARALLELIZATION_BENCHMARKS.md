# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 12:59:59 UTC | -r7gdSD2xvs_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 112.828 | 0.630 | 44.310 | 7.787 | 7.534 | 7.938 | 2.802 |

## 2026-06-27 12:59:59 UTC | -r7gdSD2xvs_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-r7gdSD2xvs_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `112.828` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.630 |
| save_clips | - |
| sample_frames | 0.749 |
| caption_frames | 29.901 |
| sample_fps | 1.964 |
| detect_object_yolo | 7.814 |
| audio_scan | 12.819 |
| asr_timings | 9.885 |
| ast_timings | 21.597 |
| describe_scenes | 7.787 |
| summarize_scenes | 7.534 |
| synthesize_synopsis | 7.938 |
| make_embedding | 2.802 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.656 |
| branch_yolo_total | 9.784 |
| branch_audio_total | 44.310 |
