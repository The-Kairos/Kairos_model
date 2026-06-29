# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 17:16:19 UTC | SSHRLfKforQ_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 104.063 | 0.785 | 31.662 | 7.999 | 17.578 | 21.146 | 1.867 |

## 2026-06-25 17:16:19 UTC | SSHRLfKforQ_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/SSHRLfKforQ_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `104.063` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.785 |
| save_clips | - |
| sample_frames | 0.308 |
| caption_frames | 13.827 |
| sample_fps | 1.827 |
| detect_object_yolo | 5.674 |
| audio_scan | 10.776 |
| asr_timings | 10.949 |
| ast_timings | 9.929 |
| describe_scenes | 7.999 |
| summarize_scenes | 17.578 |
| synthesize_synopsis | 21.146 |
| make_embedding | 1.867 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 14.141 |
| branch_yolo_total | 7.507 |
| branch_audio_total | 31.662 |
