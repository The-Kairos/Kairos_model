# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 23:35:51 UTC | _SR80jU9hPU_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 174.414 | 0.880 | 59.109 | 14.867 | 17.961 | 10.690 | 4.472 |

## 2026-06-25 23:35:51 UTC | _SR80jU9hPU_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/_SR80jU9hPU_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `174.414` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.880 |
| save_clips | - |
| sample_frames | 1.650 |
| caption_frames | 50.509 |
| sample_fps | 2.608 |
| detect_object_yolo | 10.231 |
| audio_scan | 11.918 |
| asr_timings | 7.738 |
| ast_timings | 39.445 |
| describe_scenes | 14.867 |
| summarize_scenes | 17.961 |
| synthesize_synopsis | 10.690 |
| make_embedding | 4.472 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 52.165 |
| branch_yolo_total | 12.844 |
| branch_audio_total | 59.109 |
