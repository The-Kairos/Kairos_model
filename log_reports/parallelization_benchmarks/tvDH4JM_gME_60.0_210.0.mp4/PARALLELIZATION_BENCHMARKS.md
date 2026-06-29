# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 00:10:01 UTC | tvDH4JM_gME_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 207.172 | 0.810 | 70.745 | 23.571 | 8.619 | 9.859 | 6.860 |

## 2026-06-27 00:10:01 UTC | tvDH4JM_gME_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/tvDH4JM_gME_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `207.172` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.810 |
| save_clips | - |
| sample_frames | 2.043 |
| caption_frames | 67.860 |
| sample_fps | 2.896 |
| detect_object_yolo | 12.465 |
| audio_scan | 7.576 |
| asr_timings | 10.805 |
| ast_timings | 52.355 |
| describe_scenes | 23.571 |
| summarize_scenes | 8.619 |
| synthesize_synopsis | 9.859 |
| make_embedding | 6.860 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 69.908 |
| branch_yolo_total | 15.366 |
| branch_audio_total | 70.745 |
