# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 15:03:32 UTC | lL-8C12lsu0_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 288.724 | 0.644 | 83.186 | 33.078 | 36.892 | 39.880 | 7.346 |

## 2026-06-26 15:03:32 UTC | lL-8C12lsu0_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/lL-8C12lsu0_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `288.724` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.644 |
| save_clips | - |
| sample_frames | 1.360 |
| caption_frames | 69.728 |
| sample_fps | 2.329 |
| detect_object_yolo | 12.867 |
| audio_scan | 16.111 |
| asr_timings | 15.863 |
| ast_timings | 51.203 |
| describe_scenes | 33.078 |
| summarize_scenes | 36.892 |
| synthesize_synopsis | 39.880 |
| make_embedding | 7.346 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 71.094 |
| branch_yolo_total | 15.201 |
| branch_audio_total | 83.186 |
