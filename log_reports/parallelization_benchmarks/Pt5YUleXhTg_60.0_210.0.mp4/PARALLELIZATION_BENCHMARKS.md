# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 14:56:55 UTC | Pt5YUleXhTg_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 240.553 | 0.788 | 70.781 | 26.709 | 27.901 | 36.108 | 5.408 |

## 2026-06-25 14:56:55 UTC | Pt5YUleXhTg_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Pt5YUleXhTg_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `240.553` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.788 |
| save_clips | - |
| sample_frames | 1.435 |
| caption_frames | 56.350 |
| sample_fps | 2.469 |
| detect_object_yolo | 11.145 |
| audio_scan | 15.559 |
| asr_timings | 12.553 |
| ast_timings | 42.660 |
| describe_scenes | 26.709 |
| summarize_scenes | 27.901 |
| synthesize_synopsis | 36.108 |
| make_embedding | 5.408 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 57.791 |
| branch_yolo_total | 13.620 |
| branch_audio_total | 70.781 |
