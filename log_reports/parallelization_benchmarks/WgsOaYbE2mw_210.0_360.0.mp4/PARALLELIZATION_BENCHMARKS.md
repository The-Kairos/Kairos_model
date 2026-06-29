# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 20:56:00 UTC | WgsOaYbE2mw_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 116.852 | 0.795 | 36.265 | 11.408 | 11.570 | 10.710 | 3.308 |

## 2026-06-25 20:56:00 UTC | WgsOaYbE2mw_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/WgsOaYbE2mw_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `116.852` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.795 |
| save_clips | - |
| sample_frames | 0.818 |
| caption_frames | 35.441 |
| sample_fps | 2.111 |
| detect_object_yolo | 7.884 |
| audio_scan | 3.826 |
| asr_timings | 0.000 |
| ast_timings | 27.594 |
| describe_scenes | 11.408 |
| summarize_scenes | 11.570 |
| synthesize_synopsis | 10.710 |
| make_embedding | 3.308 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.265 |
| branch_yolo_total | 10.001 |
| branch_audio_total | 31.429 |
