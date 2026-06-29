# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 18:47:46 UTC | U3QUfxqnfJg_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 205.208 | 0.664 | 63.928 | 23.153 | 20.606 | 19.764 | 5.114 |

## 2026-06-25 18:47:46 UTC | U3QUfxqnfJg_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/U3QUfxqnfJg_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `205.208` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.664 |
| save_clips | - |
| sample_frames | 1.200 |
| caption_frames | 56.532 |
| sample_fps | 2.223 |
| detect_object_yolo | 10.633 |
| audio_scan | 8.586 |
| asr_timings | 13.173 |
| ast_timings | 42.161 |
| describe_scenes | 23.153 |
| summarize_scenes | 20.606 |
| synthesize_synopsis | 19.764 |
| make_embedding | 5.114 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 57.737 |
| branch_yolo_total | 12.861 |
| branch_audio_total | 63.928 |
