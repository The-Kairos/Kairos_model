# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 08:01:55 UTC | i9scCMPwu8I_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 229.198 | 0.689 | 65.170 | 32.456 | 29.767 | 25.948 | 5.064 |

## 2026-06-26 08:01:55 UTC | i9scCMPwu8I_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/i9scCMPwu8I_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `229.198` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.689 |
| save_clips | - |
| sample_frames | 1.893 |
| caption_frames | 53.426 |
| sample_fps | 2.514 |
| detect_object_yolo | 10.823 |
| audio_scan | 14.042 |
| asr_timings | 10.345 |
| ast_timings | 40.775 |
| describe_scenes | 32.456 |
| summarize_scenes | 29.767 |
| synthesize_synopsis | 25.948 |
| make_embedding | 5.064 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 55.326 |
| branch_yolo_total | 13.343 |
| branch_audio_total | 65.170 |
