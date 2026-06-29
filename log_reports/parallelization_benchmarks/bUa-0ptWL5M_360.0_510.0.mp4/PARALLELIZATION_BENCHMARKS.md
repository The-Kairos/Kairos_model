# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 01:02:07 UTC | bUa-0ptWL5M_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 136.213 | 0.783 | 50.131 | 7.967 | 18.969 | 10.808 | 2.987 |

## 2026-06-26 01:02:07 UTC | bUa-0ptWL5M_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/bUa-0ptWL5M_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `136.213` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.783 |
| save_clips | - |
| sample_frames | 0.852 |
| caption_frames | 31.766 |
| sample_fps | 2.169 |
| detect_object_yolo | 8.375 |
| audio_scan | 15.915 |
| asr_timings | 10.007 |
| ast_timings | 24.202 |
| describe_scenes | 7.967 |
| summarize_scenes | 18.969 |
| synthesize_synopsis | 10.808 |
| make_embedding | 2.987 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.624 |
| branch_yolo_total | 10.550 |
| branch_audio_total | 50.131 |
