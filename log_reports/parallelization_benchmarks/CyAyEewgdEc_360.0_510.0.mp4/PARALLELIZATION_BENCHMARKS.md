# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 21:42:13 UTC | CyAyEewgdEc_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 324.777 | 0.797 | 228.998 | 11.577 | 18.002 | 9.599 | 3.367 |

## 2026-06-24 21:42:13 UTC | CyAyEewgdEc_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/CyAyEewgdEc_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `324.777` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.797 |
| save_clips | - |
| sample_frames | 1.241 |
| caption_frames | 38.818 |
| sample_fps | 2.321 |
| detect_object_yolo | 8.669 |
| audio_scan | 15.077 |
| asr_timings | 186.920 |
| ast_timings | 26.992 |
| describe_scenes | 11.577 |
| summarize_scenes | 18.002 |
| synthesize_synopsis | 9.599 |
| make_embedding | 3.367 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.066 |
| branch_yolo_total | 10.996 |
| branch_audio_total | 228.998 |
