# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 06:00:53 UTC | zeWShkauzL0_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 182.521 | 0.721 | 66.473 | 12.775 | 15.072 | 8.982 | 5.386 |

## 2026-06-27 06:00:53 UTC | zeWShkauzL0_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/zeWShkauzL0_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `182.521` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.721 |
| save_clips | - |
| sample_frames | 1.793 |
| caption_frames | 55.704 |
| sample_fps | 2.560 |
| detect_object_yolo | 11.559 |
| audio_scan | 12.966 |
| asr_timings | 9.124 |
| ast_timings | 44.374 |
| describe_scenes | 12.775 |
| summarize_scenes | 15.072 |
| synthesize_synopsis | 8.982 |
| make_embedding | 5.386 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 57.504 |
| branch_yolo_total | 14.124 |
| branch_audio_total | 66.473 |
