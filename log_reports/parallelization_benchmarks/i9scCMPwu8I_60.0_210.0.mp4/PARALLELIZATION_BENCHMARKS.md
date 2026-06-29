# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 07:58:04 UTC | i9scCMPwu8I_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 167.336 | 0.692 | 47.920 | 26.211 | 27.090 | 16.675 | 3.133 |

## 2026-06-26 07:58:04 UTC | i9scCMPwu8I_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/i9scCMPwu8I_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `167.336` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.692 |
| save_clips | - |
| sample_frames | 0.995 |
| caption_frames | 33.229 |
| sample_fps | 2.078 |
| detect_object_yolo | 7.854 |
| audio_scan | 14.145 |
| asr_timings | 9.027 |
| ast_timings | 24.739 |
| describe_scenes | 26.211 |
| summarize_scenes | 27.090 |
| synthesize_synopsis | 16.675 |
| make_embedding | 3.133 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.230 |
| branch_yolo_total | 9.938 |
| branch_audio_total | 47.920 |
