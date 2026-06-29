# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 05:47:36 UTC | gnsiIPjG3hk_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 215.447 | 0.791 | 66.067 | 20.877 | 21.502 | 18.841 | 6.082 |

## 2026-06-26 05:47:36 UTC | gnsiIPjG3hk_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/gnsiIPjG3hk_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `215.447` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.791 |
| save_clips | - |
| sample_frames | 1.489 |
| caption_frames | 64.082 |
| sample_fps | 2.576 |
| detect_object_yolo | 11.730 |
| audio_scan | 8.587 |
| asr_timings | 6.878 |
| ast_timings | 50.587 |
| describe_scenes | 20.877 |
| summarize_scenes | 21.502 |
| synthesize_synopsis | 18.841 |
| make_embedding | 6.082 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 65.577 |
| branch_yolo_total | 14.312 |
| branch_audio_total | 66.067 |
