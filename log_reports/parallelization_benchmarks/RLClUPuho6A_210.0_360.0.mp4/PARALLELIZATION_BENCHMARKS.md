# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 16:26:33 UTC | RLClUPuho6A_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 86.935 | 0.851 | 25.646 | 12.295 | 10.096 | 16.969 | 1.584 |

## 2026-06-25 16:26:33 UTC | RLClUPuho6A_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/RLClUPuho6A_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `86.935` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.851 |
| save_clips | - |
| sample_frames | 0.204 |
| caption_frames | 10.260 |
| sample_fps | 1.810 |
| detect_object_yolo | 5.812 |
| audio_scan | 7.801 |
| asr_timings | 10.585 |
| ast_timings | 7.250 |
| describe_scenes | 12.295 |
| summarize_scenes | 10.096 |
| synthesize_synopsis | 16.969 |
| make_embedding | 1.584 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 10.470 |
| branch_yolo_total | 7.629 |
| branch_audio_total | 25.646 |
