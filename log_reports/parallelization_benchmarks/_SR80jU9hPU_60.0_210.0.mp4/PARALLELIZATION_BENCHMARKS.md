# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 23:41:45 UTC | _SR80jU9hPU_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 175.053 | 0.822 | 57.395 | 18.193 | 21.387 | 11.609 | 4.215 |

## 2026-06-25 23:41:45 UTC | _SR80jU9hPU_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/_SR80jU9hPU_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `175.053` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.822 |
| save_clips | - |
| sample_frames | 1.786 |
| caption_frames | 46.286 |
| sample_fps | 2.497 |
| detect_object_yolo | 9.451 |
| audio_scan | 13.994 |
| asr_timings | 8.317 |
| ast_timings | 35.076 |
| describe_scenes | 18.193 |
| summarize_scenes | 21.387 |
| synthesize_synopsis | 11.609 |
| make_embedding | 4.215 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.078 |
| branch_yolo_total | 11.954 |
| branch_audio_total | 57.395 |
