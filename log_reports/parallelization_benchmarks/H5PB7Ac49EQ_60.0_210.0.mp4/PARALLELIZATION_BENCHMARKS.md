# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 02:32:23 UTC | H5PB7Ac49EQ_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 161.288 | 0.838 | 51.312 | 16.712 | 11.061 | 17.038 | 4.209 |

## 2026-06-25 02:32:23 UTC | H5PB7Ac49EQ_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/H5PB7Ac49EQ_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `161.288` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.838 |
| save_clips | - |
| sample_frames | 1.850 |
| caption_frames | 44.531 |
| sample_fps | 2.520 |
| detect_object_yolo | 9.809 |
| audio_scan | 6.427 |
| asr_timings | 9.303 |
| ast_timings | 35.574 |
| describe_scenes | 16.712 |
| summarize_scenes | 11.061 |
| synthesize_synopsis | 17.038 |
| make_embedding | 4.209 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.387 |
| branch_yolo_total | 12.334 |
| branch_audio_total | 51.312 |
