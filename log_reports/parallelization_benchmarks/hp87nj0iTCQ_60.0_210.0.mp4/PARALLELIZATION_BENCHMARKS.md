# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 07:02:25 UTC | hp87nj0iTCQ_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 133.549 | 0.801 | 48.900 | 11.563 | 11.354 | 16.016 | 2.860 |

## 2026-06-26 07:02:25 UTC | hp87nj0iTCQ_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/hp87nj0iTCQ_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `133.549` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.801 |
| save_clips | - |
| sample_frames | 0.744 |
| caption_frames | 30.065 |
| sample_fps | 2.082 |
| detect_object_yolo | 7.776 |
| audio_scan | 16.110 |
| asr_timings | 11.522 |
| ast_timings | 21.259 |
| describe_scenes | 11.563 |
| summarize_scenes | 11.354 |
| synthesize_synopsis | 16.016 |
| make_embedding | 2.860 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.815 |
| branch_yolo_total | 9.864 |
| branch_audio_total | 48.900 |
