# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 18:33:44 UTC | U2Svw1Y9X3E_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 221.712 | 0.814 | 67.022 | 14.693 | 51.198 | 14.924 | 5.040 |

## 2026-06-25 18:33:44 UTC | U2Svw1Y9X3E_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/U2Svw1Y9X3E_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `221.712` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.814 |
| save_clips | - |
| sample_frames | 1.560 |
| caption_frames | 52.206 |
| sample_fps | 2.434 |
| detect_object_yolo | 10.414 |
| audio_scan | 15.937 |
| asr_timings | 9.947 |
| ast_timings | 41.129 |
| describe_scenes | 14.693 |
| summarize_scenes | 51.198 |
| synthesize_synopsis | 14.924 |
| make_embedding | 5.040 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.771 |
| branch_yolo_total | 12.854 |
| branch_audio_total | 67.022 |
