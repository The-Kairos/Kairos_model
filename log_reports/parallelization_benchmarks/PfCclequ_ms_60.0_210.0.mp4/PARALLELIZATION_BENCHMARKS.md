# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 14:29:49 UTC | PfCclequ_ms_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 164.587 | 0.786 | 53.767 | 20.682 | 11.061 | 20.658 | 3.774 |

## 2026-06-25 14:29:49 UTC | PfCclequ_ms_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/PfCclequ_ms_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `164.587` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.786 |
| save_clips | - |
| sample_frames | 0.998 |
| caption_frames | 40.097 |
| sample_fps | 2.233 |
| detect_object_yolo | 9.089 |
| audio_scan | 14.476 |
| asr_timings | 9.038 |
| ast_timings | 30.245 |
| describe_scenes | 20.682 |
| summarize_scenes | 11.061 |
| synthesize_synopsis | 20.658 |
| make_embedding | 3.774 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.101 |
| branch_yolo_total | 11.328 |
| branch_audio_total | 53.767 |
