# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 05:43:58 UTC | zeCCSnfLZD4_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 112.781 | 0.769 | 44.268 | 7.384 | 5.438 | 12.940 | 2.843 |

## 2026-06-27 05:43:58 UTC | zeCCSnfLZD4_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/zeCCSnfLZD4_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `112.781` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.769 |
| save_clips | - |
| sample_frames | 0.681 |
| caption_frames | 27.584 |
| sample_fps | 2.039 |
| detect_object_yolo | 7.437 |
| audio_scan | 11.796 |
| asr_timings | 11.370 |
| ast_timings | 21.092 |
| describe_scenes | 7.384 |
| summarize_scenes | 5.438 |
| synthesize_synopsis | 12.940 |
| make_embedding | 2.843 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 28.271 |
| branch_yolo_total | 9.481 |
| branch_audio_total | 44.268 |
