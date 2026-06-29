# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 22:52:40 UTC | 4IYQN95j6ok_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 73.836 | 0.811 | 34.458 | 4.031 | 5.465 | 5.618 | 1.561 |

## 2026-06-21 22:52:40 UTC | 4IYQN95j6ok_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4IYQN95j6ok_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `73.836` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.811 |
| save_clips | - |
| sample_frames | 0.246 |
| caption_frames | 11.777 |
| sample_fps | 1.839 |
| detect_object_yolo | 6.633 |
| audio_scan | 14.973 |
| asr_timings | 12.366 |
| ast_timings | 7.111 |
| describe_scenes | 4.031 |
| summarize_scenes | 5.465 |
| synthesize_synopsis | 5.618 |
| make_embedding | 1.561 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 12.029 |
| branch_yolo_total | 8.478 |
| branch_audio_total | 34.458 |
