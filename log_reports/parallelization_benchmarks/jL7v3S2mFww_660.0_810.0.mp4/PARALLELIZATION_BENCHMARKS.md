# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 10:52:43 UTC | jL7v3S2mFww_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 191.799 | 0.684 | 53.696 | 25.885 | 24.124 | 25.097 | 3.847 |

## 2026-06-26 10:52:43 UTC | jL7v3S2mFww_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jL7v3S2mFww_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `191.799` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.684 |
| save_clips | - |
| sample_frames | 1.372 |
| caption_frames | 43.827 |
| sample_fps | 2.248 |
| detect_object_yolo | 9.611 |
| audio_scan | 8.653 |
| asr_timings | 12.184 |
| ast_timings | 32.851 |
| describe_scenes | 25.885 |
| summarize_scenes | 24.124 |
| synthesize_synopsis | 25.097 |
| make_embedding | 3.847 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.205 |
| branch_yolo_total | 11.865 |
| branch_audio_total | 53.696 |
