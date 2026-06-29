# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 02:54:59 UTC | wFbRG1IrNz0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 115.780 | 0.798 | 45.667 | 6.859 | 4.579 | 16.795 | 2.511 |

## 2026-06-27 02:54:59 UTC | wFbRG1IrNz0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/wFbRG1IrNz0_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `115.780` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.798 |
| save_clips | - |
| sample_frames | 0.812 |
| caption_frames | 26.966 |
| sample_fps | 2.123 |
| detect_object_yolo | 7.188 |
| audio_scan | 16.338 |
| asr_timings | 10.886 |
| ast_timings | 18.435 |
| describe_scenes | 6.859 |
| summarize_scenes | 4.579 |
| synthesize_synopsis | 16.795 |
| make_embedding | 2.511 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.785 |
| branch_yolo_total | 9.317 |
| branch_audio_total | 45.667 |
