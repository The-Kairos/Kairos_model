# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 12:58:23 UTC | k4LLzwmwJS8_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 238.731 | 0.816 | 68.709 | 18.463 | 33.136 | 30.916 | 5.901 |

## 2026-06-26 12:58:23 UTC | k4LLzwmwJS8_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/k4LLzwmwJS8_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `238.731` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.816 |
| save_clips | - |
| sample_frames | 1.737 |
| caption_frames | 62.835 |
| sample_fps | 2.675 |
| detect_object_yolo | 12.115 |
| audio_scan | 11.930 |
| asr_timings | 8.742 |
| ast_timings | 48.028 |
| describe_scenes | 18.463 |
| summarize_scenes | 33.136 |
| synthesize_synopsis | 30.916 |
| make_embedding | 5.901 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 64.577 |
| branch_yolo_total | 14.795 |
| branch_audio_total | 68.709 |
