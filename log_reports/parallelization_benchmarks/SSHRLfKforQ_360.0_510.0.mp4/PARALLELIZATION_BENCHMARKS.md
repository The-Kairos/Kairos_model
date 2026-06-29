# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 17:14:34 UTC | SSHRLfKforQ_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 111.040 | 0.793 | 37.441 | 10.316 | 8.621 | 18.921 | 2.331 |

## 2026-06-25 17:14:34 UTC | SSHRLfKforQ_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/SSHRLfKforQ_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `111.040` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.793 |
| save_clips | - |
| sample_frames | 0.476 |
| caption_frames | 21.765 |
| sample_fps | 1.944 |
| detect_object_yolo | 6.803 |
| audio_scan | 11.727 |
| asr_timings | 9.967 |
| ast_timings | 15.738 |
| describe_scenes | 10.316 |
| summarize_scenes | 8.621 |
| synthesize_synopsis | 18.921 |
| make_embedding | 2.331 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 22.247 |
| branch_yolo_total | 8.752 |
| branch_audio_total | 37.441 |
