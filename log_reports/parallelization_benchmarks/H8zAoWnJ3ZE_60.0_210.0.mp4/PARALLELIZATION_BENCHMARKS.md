# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 03:09:14 UTC | H8zAoWnJ3ZE_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 56.740 | 0.851 | 14.787 | 4.996 | 4.526 | 7.813 | 1.823 |

## 2026-06-25 03:09:14 UTC | H8zAoWnJ3ZE_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/H8zAoWnJ3ZE_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `56.740` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.851 |
| save_clips | - |
| sample_frames | 0.488 |
| caption_frames | 14.293 |
| sample_fps | 2.040 |
| detect_object_yolo | 5.195 |
| audio_scan | 3.789 |
| asr_timings | 0.000 |
| ast_timings | 9.540 |
| describe_scenes | 4.996 |
| summarize_scenes | 4.526 |
| synthesize_synopsis | 7.813 |
| make_embedding | 1.823 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 14.787 |
| branch_yolo_total | 7.241 |
| branch_audio_total | 13.337 |
