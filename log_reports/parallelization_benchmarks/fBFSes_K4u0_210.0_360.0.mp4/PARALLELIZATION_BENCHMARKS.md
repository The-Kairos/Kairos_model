# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 03:56:43 UTC | fBFSes_K4u0_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 128.421 | 0.881 | 59.790 | 7.911 | 9.613 | 11.231 | 2.510 |

## 2026-06-26 03:56:43 UTC | fBFSes_K4u0_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/fBFSes_K4u0_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `128.421` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.881 |
| save_clips | - |
| sample_frames | 0.861 |
| caption_frames | 24.291 |
| sample_fps | 2.097 |
| detect_object_yolo | 7.830 |
| audio_scan | 11.955 |
| asr_timings | 29.224 |
| ast_timings | 18.603 |
| describe_scenes | 7.911 |
| summarize_scenes | 9.613 |
| synthesize_synopsis | 11.231 |
| make_embedding | 2.510 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.158 |
| branch_yolo_total | 9.933 |
| branch_audio_total | 59.790 |
