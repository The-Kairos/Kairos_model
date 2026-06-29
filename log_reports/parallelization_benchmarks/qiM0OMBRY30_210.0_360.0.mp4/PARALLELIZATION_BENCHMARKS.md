# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 08:56:39 UTC | qiM0OMBRY30_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 164.032 | 0.782 | 93.199 | 8.018 | 5.851 | 8.865 | 2.950 |

## 2026-06-28 08:56:39 UTC | qiM0OMBRY30_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/qiM0OMBRY30_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `164.032` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.782 |
| save_clips | - |
| sample_frames | 1.011 |
| caption_frames | 31.613 |
| sample_fps | 2.196 |
| detect_object_yolo | 8.126 |
| audio_scan | 14.989 |
| asr_timings | 53.875 |
| ast_timings | 24.325 |
| describe_scenes | 8.018 |
| summarize_scenes | 5.851 |
| synthesize_synopsis | 8.865 |
| make_embedding | 2.950 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.630 |
| branch_yolo_total | 10.328 |
| branch_audio_total | 93.199 |
