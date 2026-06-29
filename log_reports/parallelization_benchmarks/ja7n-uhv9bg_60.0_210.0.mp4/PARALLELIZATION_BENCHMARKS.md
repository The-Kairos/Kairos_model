# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 11:23:16 UTC | ja7n-uhv9bg_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 159.878 | 0.805 | 53.409 | 15.170 | 17.137 | 15.109 | 3.585 |

## 2026-06-26 11:23:16 UTC | ja7n-uhv9bg_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ja7n-uhv9bg_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `159.878` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.805 |
| save_clips | - |
| sample_frames | 1.322 |
| caption_frames | 40.571 |
| sample_fps | 2.364 |
| detect_object_yolo | 8.977 |
| audio_scan | 12.924 |
| asr_timings | 10.191 |
| ast_timings | 30.286 |
| describe_scenes | 15.170 |
| summarize_scenes | 17.137 |
| synthesize_synopsis | 15.109 |
| make_embedding | 3.585 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.899 |
| branch_yolo_total | 11.346 |
| branch_audio_total | 53.409 |
