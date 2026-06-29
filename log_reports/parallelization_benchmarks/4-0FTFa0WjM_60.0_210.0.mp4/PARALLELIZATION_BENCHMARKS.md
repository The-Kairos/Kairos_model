# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 22:34:39 UTC | 4-0FTFa0WjM_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 121.801 | 0.632 | 50.060 | 7.102 | 5.727 | 6.017 | 3.288 |

## 2026-06-21 22:34:39 UTC | 4-0FTFa0WjM_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4-0FTFa0WjM_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `121.801` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.632 |
| save_clips | - |
| sample_frames | 1.220 |
| caption_frames | 35.692 |
| sample_fps | 2.092 |
| detect_object_yolo | 8.579 |
| audio_scan | 12.878 |
| asr_timings | 9.850 |
| ast_timings | 27.324 |
| describe_scenes | 7.102 |
| summarize_scenes | 5.727 |
| synthesize_synopsis | 6.017 |
| make_embedding | 3.288 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.918 |
| branch_yolo_total | 10.677 |
| branch_audio_total | 50.060 |
