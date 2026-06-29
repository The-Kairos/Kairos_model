# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 07:19:03 UTC | hpBZbgdZJEo_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 149.920 | 0.817 | 49.051 | 12.583 | 21.114 | 26.969 | 2.594 |

## 2026-06-26 07:19:03 UTC | hpBZbgdZJEo_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/hpBZbgdZJEo_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `149.920` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.817 |
| save_clips | - |
| sample_frames | 0.728 |
| caption_frames | 25.454 |
| sample_fps | 2.085 |
| detect_object_yolo | 7.118 |
| audio_scan | 15.106 |
| asr_timings | 15.587 |
| ast_timings | 18.348 |
| describe_scenes | 12.583 |
| summarize_scenes | 21.114 |
| synthesize_synopsis | 26.969 |
| make_embedding | 2.594 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.188 |
| branch_yolo_total | 9.209 |
| branch_audio_total | 49.051 |
