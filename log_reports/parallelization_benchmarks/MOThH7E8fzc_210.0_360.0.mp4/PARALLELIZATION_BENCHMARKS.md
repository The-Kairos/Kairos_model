# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 08:56:16 UTC | MOThH7E8fzc_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 2066.147 | 0.696 | 1959.141 | 16.862 | 11.032 | 17.138 | 3.635 |

## 2026-06-25 08:56:16 UTC | MOThH7E8fzc_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/MOThH7E8fzc_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `2066.147` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.696 |
| save_clips | - |
| sample_frames | 1.388 |
| caption_frames | 43.321 |
| sample_fps | 2.240 |
| detect_object_yolo | 9.255 |
| audio_scan | 15.956 |
| asr_timings | 1913.201 |
| ast_timings | 29.976 |
| describe_scenes | 16.862 |
| summarize_scenes | 11.032 |
| synthesize_synopsis | 17.138 |
| make_embedding | 3.635 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.715 |
| branch_yolo_total | 11.500 |
| branch_audio_total | 1959.141 |
