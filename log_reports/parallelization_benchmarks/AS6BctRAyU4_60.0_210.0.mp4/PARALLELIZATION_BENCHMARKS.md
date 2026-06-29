# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 19:02:06 UTC | AS6BctRAyU4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 198.093 | 0.767 | 68.526 | 21.781 | 10.618 | 16.157 | 5.679 |

## 2026-06-24 19:02:06 UTC | AS6BctRAyU4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/AS6BctRAyU4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `198.093` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.767 |
| save_clips | - |
| sample_frames | 1.277 |
| caption_frames | 58.012 |
| sample_fps | 2.436 |
| detect_object_yolo | 11.425 |
| audio_scan | 15.968 |
| asr_timings | 8.180 |
| ast_timings | 44.369 |
| describe_scenes | 21.781 |
| summarize_scenes | 10.618 |
| synthesize_synopsis | 16.157 |
| make_embedding | 5.679 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 59.295 |
| branch_yolo_total | 13.866 |
| branch_audio_total | 68.526 |
