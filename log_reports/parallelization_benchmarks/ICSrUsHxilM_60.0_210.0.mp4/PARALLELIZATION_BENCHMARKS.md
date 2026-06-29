# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 04:13:59 UTC | ICSrUsHxilM_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 179.703 | 0.786 | 63.100 | 16.139 | 15.343 | 12.433 | 4.682 |

## 2026-06-25 04:13:59 UTC | ICSrUsHxilM_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ICSrUsHxilM_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `179.703` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.786 |
| save_clips | - |
| sample_frames | 1.430 |
| caption_frames | 51.606 |
| sample_fps | 2.441 |
| detect_object_yolo | 10.353 |
| audio_scan | 12.772 |
| asr_timings | 12.838 |
| ast_timings | 37.481 |
| describe_scenes | 16.139 |
| summarize_scenes | 15.343 |
| synthesize_synopsis | 12.433 |
| make_embedding | 4.682 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.042 |
| branch_yolo_total | 12.800 |
| branch_audio_total | 63.100 |
