# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 14:06:18 UTC | 7Q3Gpf51QFU_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 231.592 | 0.796 | 66.943 | 22.500 | 25.941 | 41.666 | 5.017 |

## 2026-06-24 14:06:18 UTC | 7Q3Gpf51QFU_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/7Q3Gpf51QFU_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `231.592` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.796 |
| save_clips | - |
| sample_frames | 1.281 |
| caption_frames | 53.171 |
| sample_fps | 2.400 |
| detect_object_yolo | 10.483 |
| audio_scan | 14.859 |
| asr_timings | 12.182 |
| ast_timings | 39.893 |
| describe_scenes | 22.500 |
| summarize_scenes | 25.941 |
| synthesize_synopsis | 41.666 |
| make_embedding | 5.017 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.458 |
| branch_yolo_total | 12.889 |
| branch_audio_total | 66.943 |
