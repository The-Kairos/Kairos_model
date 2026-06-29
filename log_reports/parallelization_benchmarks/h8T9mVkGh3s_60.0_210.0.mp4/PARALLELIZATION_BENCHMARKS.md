# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 06:00:03 UTC | h8T9mVkGh3s_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 232.976 | 0.640 | 68.450 | 26.576 | 27.265 | 26.522 | 5.775 |

## 2026-06-26 06:00:03 UTC | h8T9mVkGh3s_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/h8T9mVkGh3s_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `232.976` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.640 |
| save_clips | - |
| sample_frames | 1.359 |
| caption_frames | 61.332 |
| sample_fps | 2.365 |
| detect_object_yolo | 11.290 |
| audio_scan | 12.866 |
| asr_timings | 7.973 |
| ast_timings | 47.602 |
| describe_scenes | 26.576 |
| summarize_scenes | 27.265 |
| synthesize_synopsis | 26.522 |
| make_embedding | 5.775 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 62.697 |
| branch_yolo_total | 13.660 |
| branch_audio_total | 68.450 |
