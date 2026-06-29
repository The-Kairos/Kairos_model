# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 03:56:33 UTC | xird09yJOJ4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 101.594 | 0.790 | 41.760 | 5.749 | 5.871 | 7.999 | 2.559 |

## 2026-06-27 03:56:33 UTC | xird09yJOJ4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/xird09yJOJ4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `101.594` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.790 |
| save_clips | - |
| sample_frames | 0.663 |
| caption_frames | 25.468 |
| sample_fps | 2.098 |
| detect_object_yolo | 7.219 |
| audio_scan | 14.094 |
| asr_timings | 8.936 |
| ast_timings | 18.721 |
| describe_scenes | 5.749 |
| summarize_scenes | 5.871 |
| synthesize_synopsis | 7.999 |
| make_embedding | 2.559 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.137 |
| branch_yolo_total | 9.323 |
| branch_audio_total | 41.760 |
