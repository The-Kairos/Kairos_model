# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 06:35:04 UTC | KPtayuu0L8Y_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 107.081 | 0.745 | 36.012 | 7.587 | 18.467 | 16.767 | 1.890 |

## 2026-06-25 06:35:04 UTC | KPtayuu0L8Y_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/KPtayuu0L8Y_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `107.081` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.745 |
| save_clips | - |
| sample_frames | 0.358 |
| caption_frames | 15.828 |
| sample_fps | 1.761 |
| detect_object_yolo | 6.258 |
| audio_scan | 15.952 |
| asr_timings | 9.992 |
| ast_timings | 10.059 |
| describe_scenes | 7.587 |
| summarize_scenes | 18.467 |
| synthesize_synopsis | 16.767 |
| make_embedding | 1.890 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 16.193 |
| branch_yolo_total | 8.025 |
| branch_audio_total | 36.012 |
