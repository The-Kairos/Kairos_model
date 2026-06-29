# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 17:18:01 UTC | SSHRLfKforQ_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 101.139 | 0.767 | 36.763 | 7.585 | 5.048 | 21.086 | 2.085 |

## 2026-06-25 17:18:01 UTC | SSHRLfKforQ_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/SSHRLfKforQ_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `101.139` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.767 |
| save_clips | - |
| sample_frames | 0.269 |
| caption_frames | 17.316 |
| sample_fps | 1.849 |
| detect_object_yolo | 6.980 |
| audio_scan | 12.968 |
| asr_timings | 11.428 |
| ast_timings | 12.358 |
| describe_scenes | 7.585 |
| summarize_scenes | 5.048 |
| synthesize_synopsis | 21.086 |
| make_embedding | 2.085 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 17.591 |
| branch_yolo_total | 8.836 |
| branch_audio_total | 36.763 |
