# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 07:49:52 UTC | i9scCMPwu8I_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 143.127 | 0.684 | 42.150 | 15.323 | 20.270 | 26.608 | 2.505 |

## 2026-06-26 07:49:52 UTC | i9scCMPwu8I_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/i9scCMPwu8I_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `143.127` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.684 |
| save_clips | - |
| sample_frames | 0.792 |
| caption_frames | 24.347 |
| sample_fps | 1.937 |
| detect_object_yolo | 7.084 |
| audio_scan | 15.125 |
| asr_timings | 9.198 |
| ast_timings | 17.819 |
| describe_scenes | 15.323 |
| summarize_scenes | 20.270 |
| synthesize_synopsis | 26.608 |
| make_embedding | 2.505 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.145 |
| branch_yolo_total | 9.027 |
| branch_audio_total | 42.150 |
