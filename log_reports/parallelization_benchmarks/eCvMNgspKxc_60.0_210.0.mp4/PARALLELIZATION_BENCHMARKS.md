# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 03:30:58 UTC | eCvMNgspKxc_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 112.043 | 0.705 | 39.914 | 7.474 | 8.392 | 9.047 | 2.998 |

## 2026-06-26 03:30:58 UTC | eCvMNgspKxc_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/eCvMNgspKxc_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `112.043` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.705 |
| save_clips | - |
| sample_frames | 0.987 |
| caption_frames | 31.175 |
| sample_fps | 2.033 |
| detect_object_yolo | 7.923 |
| audio_scan | 7.611 |
| asr_timings | 7.755 |
| ast_timings | 24.539 |
| describe_scenes | 7.474 |
| summarize_scenes | 8.392 |
| synthesize_synopsis | 9.047 |
| make_embedding | 2.998 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.168 |
| branch_yolo_total | 9.962 |
| branch_audio_total | 39.914 |
