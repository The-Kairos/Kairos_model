# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 13:23:48 UTC | 1IDUWll4TXo_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 208.333 | 0.810 | 56.732 | 18.909 | 29.994 | 33.216 | 4.226 |
| 2026-06-27 14:59:41 UTC | 1IDUWll4TXo_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 152.831 | 0.820 | 58.890 | 11.282 | 7.088 | 6.804 | 4.205 |

## 2026-06-23 13:23:48 UTC | 1IDUWll4TXo_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1IDUWll4TXo_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `208.333` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.810 |
| save_clips | - |
| sample_frames | 1.762 |
| caption_frames | 49.247 |
| sample_fps | 2.541 |
| detect_object_yolo | 9.467 |
| audio_scan | 14.842 |
| asr_timings | 7.645 |
| ast_timings | 34.237 |
| describe_scenes | 18.909 |
| summarize_scenes | 29.994 |
| synthesize_synopsis | 33.216 |
| make_embedding | 4.226 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.015 |
| branch_yolo_total | 12.015 |
| branch_audio_total | 56.732 |

## 2026-06-27 14:59:41 UTC | 1IDUWll4TXo_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1IDUWll4TXo_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `152.831` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.820 |
| save_clips | - |
| sample_frames | 1.787 |
| caption_frames | 48.714 |
| sample_fps | 2.531 |
| detect_object_yolo | 9.309 |
| audio_scan | 14.935 |
| asr_timings | 9.087 |
| ast_timings | 34.860 |
| describe_scenes | 11.282 |
| summarize_scenes | 7.088 |
| synthesize_synopsis | 6.804 |
| make_embedding | 4.205 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.507 |
| branch_yolo_total | 11.846 |
| branch_audio_total | 58.890 |
