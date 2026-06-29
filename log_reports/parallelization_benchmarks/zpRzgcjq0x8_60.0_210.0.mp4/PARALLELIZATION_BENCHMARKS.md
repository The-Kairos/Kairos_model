# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 06:17:52 UTC | zpRzgcjq0x8_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 199.284 | 0.734 | 72.635 | 16.680 | 10.944 | 6.635 | 6.048 |

## 2026-06-27 06:17:52 UTC | zpRzgcjq0x8_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/zpRzgcjq0x8_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `199.284` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.734 |
| save_clips | - |
| sample_frames | 1.954 |
| caption_frames | 67.355 |
| sample_fps | 2.728 |
| detect_object_yolo | 12.156 |
| audio_scan | 15.007 |
| asr_timings | 8.128 |
| ast_timings | 49.492 |
| describe_scenes | 16.680 |
| summarize_scenes | 10.944 |
| synthesize_synopsis | 6.635 |
| make_embedding | 6.048 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 69.316 |
| branch_yolo_total | 14.890 |
| branch_audio_total | 72.635 |
