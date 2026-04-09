# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-08 09:24:00 UTC | Statistical_Learning__5.2_K-fold_Cross_Validation.mp4 | parallel | gemini | gemini-embedding-001 | 214.377 | 8.305 | 88.638 | 69.760 | 28.244 | 11.734 | 2.855 |

## 2026-04-08 09:24:00 UTC | Statistical_Learning__5.2_K-fold_Cross_Validation.mp4 | parallel

- Video path: `/var/tmp/kairos/jobs/4d0ae326-bce4-46b2-9c50-6462a7fc436b/Statistical_Learning__5.2_K-fold_Cross_Validation.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `214.377` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 8.305 |
| save_clips | - |
| sample_frames | 2.119 |
| caption_frames | 56.846 |
| sample_fps | 11.148 |
| detect_object_yolo | 29.750 |
| audio_scan | 49.629 |
| asr_timings | 38.999 |
| ast_timings | 20.595 |
| describe_scenes | 69.760 |
| summarize_scenes | 28.244 |
| synthesize_synopsis | 11.734 |
| make_embedding | 2.855 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 58.974 |
| branch_yolo_total | 40.906 |
| branch_audio_total | 88.638 |
