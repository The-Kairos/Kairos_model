# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 20:15:03 UTC | C9JnjCVIW5k_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 228.959 | 0.860 | 86.225 | 15.003 | 32.238 | 9.837 | 5.937 |

## 2026-06-24 20:15:03 UTC | C9JnjCVIW5k_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/C9JnjCVIW5k_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `228.959` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.860 |
| save_clips | - |
| sample_frames | 2.000 |
| caption_frames | 60.638 |
| sample_fps | 2.862 |
| detect_object_yolo | 11.891 |
| audio_scan | 8.705 |
| asr_timings | 30.054 |
| ast_timings | 47.457 |
| describe_scenes | 15.003 |
| summarize_scenes | 32.238 |
| synthesize_synopsis | 9.837 |
| make_embedding | 5.937 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 62.644 |
| branch_yolo_total | 14.759 |
| branch_audio_total | 86.225 |
