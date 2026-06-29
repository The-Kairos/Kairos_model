# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 14:53:53 UTC | l6aGqD9b53w_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 147.097 | 0.805 | 49.638 | 12.809 | 10.816 | 18.205 | 3.322 |

## 2026-06-26 14:53:53 UTC | l6aGqD9b53w_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/l6aGqD9b53w_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `147.097` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.805 |
| save_clips | - |
| sample_frames | 0.829 |
| caption_frames | 38.195 |
| sample_fps | 2.171 |
| detect_object_yolo | 8.906 |
| audio_scan | 11.903 |
| asr_timings | 10.222 |
| ast_timings | 27.503 |
| describe_scenes | 12.809 |
| summarize_scenes | 10.816 |
| synthesize_synopsis | 18.205 |
| make_embedding | 3.322 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.029 |
| branch_yolo_total | 11.082 |
| branch_audio_total | 49.638 |
