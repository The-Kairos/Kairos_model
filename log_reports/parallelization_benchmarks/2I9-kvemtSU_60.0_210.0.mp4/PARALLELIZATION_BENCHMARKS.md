# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 21:12:16 UTC | 2I9-kvemtSU_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 113.422 | 0.780 | 43.210 | 10.003 | 5.293 | 7.857 | 3.027 |

## 2026-06-21 21:12:16 UTC | 2I9-kvemtSU_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2I9-kvemtSU_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `113.422` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.780 |
| save_clips | - |
| sample_frames | 0.878 |
| caption_frames | 31.119 |
| sample_fps | 2.106 |
| detect_object_yolo | 7.741 |
| audio_scan | 6.472 |
| asr_timings | 12.960 |
| ast_timings | 23.770 |
| describe_scenes | 10.003 |
| summarize_scenes | 5.293 |
| synthesize_synopsis | 7.857 |
| make_embedding | 3.027 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.003 |
| branch_yolo_total | 9.853 |
| branch_audio_total | 43.210 |
