# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 17:12:42 UTC | SSHRLfKforQ_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 124.511 | 0.774 | 42.707 | 14.549 | 10.910 | 16.252 | 2.623 |

## 2026-06-25 17:12:42 UTC | SSHRLfKforQ_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/SSHRLfKforQ_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `124.511` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.774 |
| save_clips | - |
| sample_frames | 0.619 |
| caption_frames | 25.319 |
| sample_fps | 2.020 |
| detect_object_yolo | 7.277 |
| audio_scan | 14.798 |
| asr_timings | 9.489 |
| ast_timings | 18.412 |
| describe_scenes | 14.549 |
| summarize_scenes | 10.910 |
| synthesize_synopsis | 16.252 |
| make_embedding | 2.623 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.944 |
| branch_yolo_total | 9.302 |
| branch_audio_total | 42.707 |
