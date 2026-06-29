# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 07:48:16 UTC | LmnfadjuaE8_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 182.944 | 0.640 | 52.505 | 18.450 | 36.873 | 20.236 | 3.379 |

## 2026-06-25 07:48:16 UTC | LmnfadjuaE8_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/LmnfadjuaE8_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `182.944` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.640 |
| save_clips | - |
| sample_frames | 1.128 |
| caption_frames | 37.259 |
| sample_fps | 2.148 |
| detect_object_yolo | 8.819 |
| audio_scan | 15.158 |
| asr_timings | 10.420 |
| ast_timings | 26.918 |
| describe_scenes | 18.450 |
| summarize_scenes | 36.873 |
| synthesize_synopsis | 20.236 |
| make_embedding | 3.379 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.392 |
| branch_yolo_total | 10.973 |
| branch_audio_total | 52.505 |
