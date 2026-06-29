# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 11:59:27 UTC | OOJnQDyM5Us_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 164.283 | 0.800 | 56.580 | 14.540 | 17.804 | 26.957 | 2.794 |

## 2026-06-25 11:59:27 UTC | OOJnQDyM5Us_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/OOJnQDyM5Us_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `164.283` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.800 |
| save_clips | - |
| sample_frames | 0.946 |
| caption_frames | 31.954 |
| sample_fps | 2.180 |
| detect_object_yolo | 8.242 |
| audio_scan | 15.774 |
| asr_timings | 19.003 |
| ast_timings | 21.794 |
| describe_scenes | 14.540 |
| summarize_scenes | 17.804 |
| synthesize_synopsis | 26.957 |
| make_embedding | 2.794 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.906 |
| branch_yolo_total | 10.429 |
| branch_audio_total | 56.580 |
