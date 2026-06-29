# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 18:52:55 UTC | rXZfBpQq3I8_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 226.941 | 0.791 | 68.177 | 25.504 | 37.194 | 15.472 | 5.434 |

## 2026-06-26 18:52:55 UTC | rXZfBpQq3I8_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/rXZfBpQq3I8_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `226.941` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.791 |
| save_clips | - |
| sample_frames | 1.430 |
| caption_frames | 57.644 |
| sample_fps | 2.501 |
| detect_object_yolo | 11.332 |
| audio_scan | 12.726 |
| asr_timings | 11.542 |
| ast_timings | 43.900 |
| describe_scenes | 25.504 |
| summarize_scenes | 37.194 |
| synthesize_synopsis | 15.472 |
| make_embedding | 5.434 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 59.080 |
| branch_yolo_total | 13.838 |
| branch_audio_total | 68.177 |
