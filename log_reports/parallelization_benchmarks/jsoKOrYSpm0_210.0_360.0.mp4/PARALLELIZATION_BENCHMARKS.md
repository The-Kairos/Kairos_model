# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 12:23:16 UTC | jsoKOrYSpm0_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 215.807 | 0.794 | 63.506 | 17.669 | 26.773 | 30.130 | 5.201 |

## 2026-06-26 12:23:16 UTC | jsoKOrYSpm0_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jsoKOrYSpm0_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `215.807` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.794 |
| save_clips | - |
| sample_frames | 1.284 |
| caption_frames | 55.660 |
| sample_fps | 2.454 |
| detect_object_yolo | 10.927 |
| audio_scan | 12.894 |
| asr_timings | 8.782 |
| ast_timings | 41.821 |
| describe_scenes | 17.669 |
| summarize_scenes | 26.773 |
| synthesize_synopsis | 30.130 |
| make_embedding | 5.201 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 56.950 |
| branch_yolo_total | 13.387 |
| branch_audio_total | 63.506 |
