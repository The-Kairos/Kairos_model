# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 16:32:22 UTC | nCsbm8Ze8oM_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 165.944 | 0.797 | 65.920 | 14.223 | 9.233 | 8.744 | 4.492 |

## 2026-06-27 16:32:22 UTC | nCsbm8Ze8oM_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/nCsbm8Ze8oM_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `165.944` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.797 |
| save_clips | - |
| sample_frames | 1.346 |
| caption_frames | 47.689 |
| sample_fps | 2.382 |
| detect_object_yolo | 9.747 |
| audio_scan | 15.868 |
| asr_timings | 12.892 |
| ast_timings | 37.151 |
| describe_scenes | 14.223 |
| summarize_scenes | 9.233 |
| synthesize_synopsis | 8.744 |
| make_embedding | 4.492 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.041 |
| branch_yolo_total | 12.134 |
| branch_audio_total | 65.920 |
