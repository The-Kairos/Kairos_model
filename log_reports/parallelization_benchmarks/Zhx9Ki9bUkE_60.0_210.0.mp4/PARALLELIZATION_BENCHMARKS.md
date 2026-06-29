# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 22:54:03 UTC | Zhx9Ki9bUkE_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 145.012 | 0.782 | 56.457 | 8.113 | 11.772 | 10.531 | 3.553 |

## 2026-06-25 22:54:03 UTC | Zhx9Ki9bUkE_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Zhx9Ki9bUkE_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `145.012` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.782 |
| save_clips | - |
| sample_frames | 0.950 |
| caption_frames | 40.736 |
| sample_fps | 2.198 |
| detect_object_yolo | 8.504 |
| audio_scan | 13.789 |
| asr_timings | 12.640 |
| ast_timings | 30.019 |
| describe_scenes | 8.113 |
| summarize_scenes | 11.772 |
| synthesize_synopsis | 10.531 |
| make_embedding | 3.553 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.692 |
| branch_yolo_total | 10.708 |
| branch_audio_total | 56.457 |
