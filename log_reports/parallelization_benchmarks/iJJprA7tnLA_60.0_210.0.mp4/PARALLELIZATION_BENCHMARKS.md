# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 08:29:27 UTC | iJJprA7tnLA_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 165.944 | 0.675 | 50.748 | 16.980 | 22.766 | 19.911 | 3.757 |

## 2026-06-26 08:29:27 UTC | iJJprA7tnLA_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/iJJprA7tnLA_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `165.944` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.675 |
| save_clips | - |
| sample_frames | 0.873 |
| caption_frames | 38.120 |
| sample_fps | 2.042 |
| detect_object_yolo | 8.660 |
| audio_scan | 8.628 |
| asr_timings | 12.046 |
| ast_timings | 30.066 |
| describe_scenes | 16.980 |
| summarize_scenes | 22.766 |
| synthesize_synopsis | 19.911 |
| make_embedding | 3.757 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.999 |
| branch_yolo_total | 10.708 |
| branch_audio_total | 50.748 |
