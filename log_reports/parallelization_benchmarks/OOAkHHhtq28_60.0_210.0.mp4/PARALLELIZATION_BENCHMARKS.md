# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 11:49:19 UTC | OOAkHHhtq28_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 237.462 | 0.676 | 68.642 | 33.803 | 23.128 | 25.504 | 5.989 |

## 2026-06-25 11:49:19 UTC | OOAkHHhtq28_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/OOAkHHhtq28_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `237.462` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.676 |
| save_clips | - |
| sample_frames | 1.977 |
| caption_frames | 62.289 |
| sample_fps | 2.504 |
| detect_object_yolo | 11.527 |
| audio_scan | 9.861 |
| asr_timings | 11.752 |
| ast_timings | 47.021 |
| describe_scenes | 33.803 |
| summarize_scenes | 23.128 |
| synthesize_synopsis | 25.504 |
| make_embedding | 5.989 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 64.272 |
| branch_yolo_total | 14.038 |
| branch_audio_total | 68.642 |
