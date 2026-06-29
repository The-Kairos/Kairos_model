# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 18:54:39 UTC | APwfWItEVks_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 133.825 | 0.774 | 44.676 | 10.881 | 9.115 | 18.435 | 3.297 |

## 2026-06-24 18:54:39 UTC | APwfWItEVks_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/APwfWItEVks_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `133.825` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.774 |
| save_clips | - |
| sample_frames | 0.789 |
| caption_frames | 34.055 |
| sample_fps | 2.147 |
| detect_object_yolo | 8.271 |
| audio_scan | 7.566 |
| asr_timings | 10.659 |
| ast_timings | 26.443 |
| describe_scenes | 10.881 |
| summarize_scenes | 9.115 |
| synthesize_synopsis | 18.435 |
| make_embedding | 3.297 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.850 |
| branch_yolo_total | 10.424 |
| branch_audio_total | 44.676 |
