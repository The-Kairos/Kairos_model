# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 23:42:55 UTC | _UxECICm2eY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 68.777 | 0.670 | 24.723 | 5.173 | 8.028 | 11.820 | 1.273 |

## 2026-06-25 23:42:55 UTC | _UxECICm2eY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/_UxECICm2eY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `68.777` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.670 |
| save_clips | - |
| sample_frames | 0.094 |
| caption_frames | 8.320 |
| sample_fps | 1.615 |
| detect_object_yolo | 5.652 |
| audio_scan | 10.674 |
| asr_timings | 9.553 |
| ast_timings | 4.488 |
| describe_scenes | 5.173 |
| summarize_scenes | 8.028 |
| synthesize_synopsis | 11.820 |
| make_embedding | 1.273 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 8.420 |
| branch_yolo_total | 7.273 |
| branch_audio_total | 24.723 |
