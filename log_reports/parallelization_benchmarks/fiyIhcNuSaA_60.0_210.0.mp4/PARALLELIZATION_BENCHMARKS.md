# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 04:38:36 UTC | fiyIhcNuSaA_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 108.551 | 0.812 | 41.203 | 7.666 | 5.716 | 17.013 | 2.330 |

## 2026-06-26 04:38:36 UTC | fiyIhcNuSaA_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/fiyIhcNuSaA_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `108.551` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.812 |
| save_clips | - |
| sample_frames | 0.693 |
| caption_frames | 22.541 |
| sample_fps | 2.050 |
| detect_object_yolo | 7.108 |
| audio_scan | 16.026 |
| asr_timings | 9.229 |
| ast_timings | 15.939 |
| describe_scenes | 7.666 |
| summarize_scenes | 5.716 |
| synthesize_synopsis | 17.013 |
| make_embedding | 2.330 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.240 |
| branch_yolo_total | 9.164 |
| branch_audio_total | 41.203 |
