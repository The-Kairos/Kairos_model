# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 20:39:59 UTC | WZr3CPCe8BI_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 163.026 | 0.629 | 62.932 | 16.549 | 10.659 | 11.938 | 3.860 |

## 2026-06-25 20:39:59 UTC | WZr3CPCe8BI_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/WZr3CPCe8BI_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `163.026` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.629 |
| save_clips | - |
| sample_frames | 1.018 |
| caption_frames | 43.149 |
| sample_fps | 2.134 |
| detect_object_yolo | 8.753 |
| audio_scan | 14.940 |
| asr_timings | 15.206 |
| ast_timings | 32.778 |
| describe_scenes | 16.549 |
| summarize_scenes | 10.659 |
| synthesize_synopsis | 11.938 |
| make_embedding | 3.860 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.173 |
| branch_yolo_total | 10.893 |
| branch_audio_total | 62.932 |
