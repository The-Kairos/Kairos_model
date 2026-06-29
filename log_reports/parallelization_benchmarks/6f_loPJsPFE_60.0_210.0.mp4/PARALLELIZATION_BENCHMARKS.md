# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 12:27:18 UTC | 6f_loPJsPFE_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 171.615 | 0.703 | 49.985 | 20.978 | 23.854 | 21.400 | 3.406 |

## 2026-06-24 12:27:18 UTC | 6f_loPJsPFE_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/6f_loPJsPFE_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `171.615` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.703 |
| save_clips | - |
| sample_frames | 1.066 |
| caption_frames | 37.763 |
| sample_fps | 2.129 |
| detect_object_yolo | 8.860 |
| audio_scan | 11.814 |
| asr_timings | 11.051 |
| ast_timings | 27.112 |
| describe_scenes | 20.978 |
| summarize_scenes | 23.854 |
| synthesize_synopsis | 21.400 |
| make_embedding | 3.406 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.835 |
| branch_yolo_total | 10.995 |
| branch_audio_total | 49.985 |
