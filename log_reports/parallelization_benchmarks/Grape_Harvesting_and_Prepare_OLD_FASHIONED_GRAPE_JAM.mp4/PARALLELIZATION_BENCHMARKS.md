# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-05-13 09:11:26 UTC | Grape_Harvesting_and_Prepare_OLD_FASHIONED_GRAPE_JAM.mp4 | parallel | gemini | gemini-embedding-001 | 724.752 | 4.310 | 367.696 | 247.784 | 51.542 | 22.315 | 25.245 |

## 2026-05-13 09:11:26 UTC | Grape_Harvesting_and_Prepare_OLD_FASHIONED_GRAPE_JAM.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/879072d1-756f-4670-b525-f750718c45f3/Grape_Harvesting_and_Prepare_OLD_FASHIONED_GRAPE_JAM.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `724.752` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 4.310 |
| save_clips | - |
| sample_frames | 11.163 |
| caption_frames | 205.640 |
| sample_fps | 16.951 |
| detect_object_yolo | 71.724 |
| audio_scan | 57.257 |
| asr_timings | 47.116 |
| ast_timings | 310.427 |
| describe_scenes | 247.784 |
| summarize_scenes | 51.542 |
| synthesize_synopsis | 22.315 |
| make_embedding | 25.245 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 216.812 |
| branch_yolo_total | 88.685 |
| branch_audio_total | 367.696 |
