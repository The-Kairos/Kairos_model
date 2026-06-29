# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 12:31:38 UTC | 6g0IdjnX9fQ_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 138.139 | 0.701 | 43.672 | 10.435 | 12.033 | 30.043 | 2.800 |

## 2026-06-24 12:31:38 UTC | 6g0IdjnX9fQ_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/6g0IdjnX9fQ_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `138.139` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.701 |
| save_clips | - |
| sample_frames | 0.615 |
| caption_frames | 27.351 |
| sample_fps | 1.937 |
| detect_object_yolo | 7.165 |
| audio_scan | 14.862 |
| asr_timings | 7.853 |
| ast_timings | 20.949 |
| describe_scenes | 10.435 |
| summarize_scenes | 12.033 |
| synthesize_synopsis | 30.043 |
| make_embedding | 2.800 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.971 |
| branch_yolo_total | 9.108 |
| branch_audio_total | 43.672 |
