# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 23:08:14 UTC | _6hnl_BrFvs_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 94.245 | 0.887 | 38.053 | 7.929 | 4.145 | 10.462 | 2.290 |

## 2026-06-25 23:08:14 UTC | _6hnl_BrFvs_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/_6hnl_BrFvs_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `94.245` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.887 |
| save_clips | - |
| sample_frames | 0.626 |
| caption_frames | 20.614 |
| sample_fps | 2.013 |
| detect_object_yolo | 5.817 |
| audio_scan | 9.423 |
| asr_timings | 12.720 |
| ast_timings | 15.902 |
| describe_scenes | 7.929 |
| summarize_scenes | 4.145 |
| synthesize_synopsis | 10.462 |
| make_embedding | 2.290 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 21.246 |
| branch_yolo_total | 7.836 |
| branch_audio_total | 38.053 |
