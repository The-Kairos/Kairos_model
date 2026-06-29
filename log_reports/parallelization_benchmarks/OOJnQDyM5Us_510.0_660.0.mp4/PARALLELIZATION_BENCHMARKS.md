# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 11:56:41 UTC | OOJnQDyM5Us_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 139.106 | 0.857 | 33.001 | 14.227 | 15.266 | 36.076 | 2.564 |

## 2026-06-25 11:56:41 UTC | OOJnQDyM5Us_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/OOJnQDyM5Us_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `139.106` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.857 |
| save_clips | - |
| sample_frames | 0.904 |
| caption_frames | 24.683 |
| sample_fps | 2.109 |
| detect_object_yolo | 7.971 |
| audio_scan | 7.760 |
| asr_timings | 5.990 |
| ast_timings | 19.242 |
| describe_scenes | 14.227 |
| summarize_scenes | 15.266 |
| synthesize_synopsis | 36.076 |
| make_embedding | 2.564 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.594 |
| branch_yolo_total | 10.087 |
| branch_audio_total | 33.001 |
