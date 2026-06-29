# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 16:42:49 UTC | Ray65ZMJzNE_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 107.710 | 0.790 | 55.291 | 7.199 | 4.123 | 14.948 | 1.839 |

## 2026-06-25 16:42:49 UTC | Ray65ZMJzNE_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Ray65ZMJzNE_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `107.710` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.790 |
| save_clips | - |
| sample_frames | 0.337 |
| caption_frames | 13.429 |
| sample_fps | 1.848 |
| detect_object_yolo | 6.491 |
| audio_scan | 16.772 |
| asr_timings | 28.422 |
| ast_timings | 10.088 |
| describe_scenes | 7.199 |
| summarize_scenes | 4.123 |
| synthesize_synopsis | 14.948 |
| make_embedding | 1.839 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 13.772 |
| branch_yolo_total | 8.345 |
| branch_audio_total | 55.291 |
