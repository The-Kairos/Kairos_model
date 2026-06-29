# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 08:42:32 UTC | q9IRmUEsz6g_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 189.306 | 0.707 | 68.850 | 18.753 | 12.237 | 9.014 | 5.096 |

## 2026-06-28 08:42:32 UTC | q9IRmUEsz6g_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/q9IRmUEsz6g_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `189.306` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.707 |
| save_clips | - |
| sample_frames | 1.759 |
| caption_frames | 57.710 |
| sample_fps | 2.506 |
| detect_object_yolo | 11.177 |
| audio_scan | 16.210 |
| asr_timings | 10.891 |
| ast_timings | 41.740 |
| describe_scenes | 18.753 |
| summarize_scenes | 12.237 |
| synthesize_synopsis | 9.014 |
| make_embedding | 5.096 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 59.476 |
| branch_yolo_total | 13.689 |
| branch_audio_total | 68.850 |
