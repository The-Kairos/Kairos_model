# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 07:34:05 UTC | LldGeCXP6RQ_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 134.853 | 0.795 | 44.133 | 9.164 | 15.595 | 26.370 | 2.645 |

## 2026-06-25 07:34:05 UTC | LldGeCXP6RQ_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/LldGeCXP6RQ_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `134.853` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.795 |
| save_clips | - |
| sample_frames | 0.673 |
| caption_frames | 25.044 |
| sample_fps | 1.993 |
| detect_object_yolo | 7.000 |
| audio_scan | 14.827 |
| asr_timings | 11.232 |
| ast_timings | 18.064 |
| describe_scenes | 9.164 |
| summarize_scenes | 15.595 |
| synthesize_synopsis | 26.370 |
| make_embedding | 2.645 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.723 |
| branch_yolo_total | 8.999 |
| branch_audio_total | 44.133 |
