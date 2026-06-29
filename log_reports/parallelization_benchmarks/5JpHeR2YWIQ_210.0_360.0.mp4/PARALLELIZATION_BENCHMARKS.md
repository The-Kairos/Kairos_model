# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 11:42:55 UTC | 5JpHeR2YWIQ_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 133.811 | 0.809 | 40.039 | 12.138 | 14.563 | 27.291 | 2.539 |

## 2026-06-24 11:42:55 UTC | 5JpHeR2YWIQ_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/5JpHeR2YWIQ_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `133.811` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.809 |
| save_clips | - |
| sample_frames | 0.660 |
| caption_frames | 25.101 |
| sample_fps | 2.065 |
| detect_object_yolo | 7.131 |
| audio_scan | 8.686 |
| asr_timings | 12.918 |
| ast_timings | 18.426 |
| describe_scenes | 12.138 |
| summarize_scenes | 14.563 |
| synthesize_synopsis | 27.291 |
| make_embedding | 2.539 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.768 |
| branch_yolo_total | 9.203 |
| branch_audio_total | 40.039 |
