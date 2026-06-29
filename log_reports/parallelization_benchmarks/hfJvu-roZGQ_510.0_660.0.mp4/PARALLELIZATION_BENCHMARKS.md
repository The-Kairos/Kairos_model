# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 06:35:24 UTC | hfJvu-roZGQ_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 222.511 | 1.224 | 97.060 | 23.563 | 16.271 | 21.266 | 3.927 |

## 2026-06-26 06:35:24 UTC | hfJvu-roZGQ_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/hfJvu-roZGQ_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `222.511` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.224 |
| save_clips | - |
| sample_frames | 0.951 |
| caption_frames | 46.061 |
| sample_fps | 0.995 |
| detect_object_yolo | 9.783 |
| audio_scan | 13.994 |
| asr_timings | 51.154 |
| ast_timings | 31.903 |
| describe_scenes | 23.563 |
| summarize_scenes | 16.271 |
| synthesize_synopsis | 21.266 |
| make_embedding | 3.927 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.019 |
| branch_yolo_total | 10.783 |
| branch_audio_total | 97.060 |
