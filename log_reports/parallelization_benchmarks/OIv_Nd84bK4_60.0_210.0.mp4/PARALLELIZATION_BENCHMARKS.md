# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 11:34:58 UTC | OIv_Nd84bK4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 252.677 | 0.786 | 75.615 | 30.099 | 38.558 | 16.155 | 6.189 |

## 2026-06-25 11:34:58 UTC | OIv_Nd84bK4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/OIv_Nd84bK4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `252.677` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.786 |
| save_clips | - |
| sample_frames | 1.838 |
| caption_frames | 66.787 |
| sample_fps | 2.730 |
| detect_object_yolo | 12.493 |
| audio_scan | 15.555 |
| asr_timings | 8.845 |
| ast_timings | 51.206 |
| describe_scenes | 30.099 |
| summarize_scenes | 38.558 |
| synthesize_synopsis | 16.155 |
| make_embedding | 6.189 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 68.631 |
| branch_yolo_total | 15.229 |
| branch_audio_total | 75.615 |
