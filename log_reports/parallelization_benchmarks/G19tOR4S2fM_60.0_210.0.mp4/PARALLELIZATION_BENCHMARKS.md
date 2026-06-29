# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 01:05:50 UTC | G19tOR4S2fM_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 168.029 | 0.717 | 56.420 | 20.066 | 8.403 | 21.205 | 3.887 |

## 2026-06-25 01:05:50 UTC | G19tOR4S2fM_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/G19tOR4S2fM_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `168.029` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.717 |
| save_clips | - |
| sample_frames | 1.176 |
| caption_frames | 43.302 |
| sample_fps | 2.173 |
| detect_object_yolo | 9.262 |
| audio_scan | 13.877 |
| asr_timings | 9.863 |
| ast_timings | 32.672 |
| describe_scenes | 20.066 |
| summarize_scenes | 8.403 |
| synthesize_synopsis | 21.205 |
| make_embedding | 3.887 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.483 |
| branch_yolo_total | 11.441 |
| branch_audio_total | 56.420 |
