# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 17:07:28 UTC | 8o0DZDwJR7U_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 176.839 | 0.836 | 60.039 | 13.898 | 12.899 | 21.829 | 4.973 |

## 2026-06-24 17:07:28 UTC | 8o0DZDwJR7U_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/8o0DZDwJR7U_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `176.839` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.836 |
| save_clips | - |
| sample_frames | 1.551 |
| caption_frames | 47.089 |
| sample_fps | 2.513 |
| detect_object_yolo | 9.801 |
| audio_scan | 11.894 |
| asr_timings | 12.309 |
| ast_timings | 35.828 |
| describe_scenes | 13.898 |
| summarize_scenes | 12.899 |
| synthesize_synopsis | 21.829 |
| make_embedding | 4.973 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.646 |
| branch_yolo_total | 12.321 |
| branch_audio_total | 60.039 |
