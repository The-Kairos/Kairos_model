# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 10:59:21 UTC | Ns9nKnuhgE0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 216.036 | 0.826 | 66.435 | 26.940 | 21.383 | 15.655 | 5.438 |

## 2026-06-25 10:59:21 UTC | Ns9nKnuhgE0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Ns9nKnuhgE0_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `216.036` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.826 |
| save_clips | - |
| sample_frames | 1.709 |
| caption_frames | 61.958 |
| sample_fps | 2.709 |
| detect_object_yolo | 11.508 |
| audio_scan | 12.931 |
| asr_timings | 8.482 |
| ast_timings | 45.014 |
| describe_scenes | 26.940 |
| summarize_scenes | 21.383 |
| synthesize_synopsis | 15.655 |
| make_embedding | 5.438 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 63.672 |
| branch_yolo_total | 14.223 |
| branch_audio_total | 66.435 |
