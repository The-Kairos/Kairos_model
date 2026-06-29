# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 16:41:03 UTC | 3waJEHGNYug_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 154.326 | 0.635 | 48.776 | 20.445 | 10.957 | 16.542 | 3.699 |
| 2026-06-24 10:36:13 UTC | 3waJEHGNYug_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 152.022 | 0.646 | 48.555 | 12.630 | 14.538 | 19.772 | 3.642 |

## 2026-06-23 16:41:03 UTC | 3waJEHGNYug_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3waJEHGNYug_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `154.326` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.635 |
| save_clips | - |
| sample_frames | 1.004 |
| caption_frames | 40.001 |
| sample_fps | 2.088 |
| detect_object_yolo | 8.766 |
| audio_scan | 11.703 |
| asr_timings | 7.104 |
| ast_timings | 29.961 |
| describe_scenes | 20.445 |
| summarize_scenes | 10.957 |
| synthesize_synopsis | 16.542 |
| make_embedding | 3.699 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.010 |
| branch_yolo_total | 10.860 |
| branch_audio_total | 48.776 |

## 2026-06-24 10:36:13 UTC | 3waJEHGNYug_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3waJEHGNYug_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `152.022` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.646 |
| save_clips | - |
| sample_frames | 1.011 |
| caption_frames | 38.996 |
| sample_fps | 2.103 |
| detect_object_yolo | 8.721 |
| audio_scan | 11.721 |
| asr_timings | 6.982 |
| ast_timings | 29.843 |
| describe_scenes | 12.630 |
| summarize_scenes | 14.538 |
| synthesize_synopsis | 19.772 |
| make_embedding | 3.642 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.012 |
| branch_yolo_total | 10.830 |
| branch_audio_total | 48.555 |
