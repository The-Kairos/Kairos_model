# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 12:39:32 UTC | Ovp584PUT_c_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 169.921 | 0.775 | 43.028 | 13.143 | 26.616 | 39.918 | 3.059 |

## 2026-06-25 12:39:32 UTC | Ovp584PUT_c_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Ovp584PUT_c_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `169.921` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.775 |
| save_clips | - |
| sample_frames | 0.692 |
| caption_frames | 31.534 |
| sample_fps | 2.097 |
| detect_object_yolo | 7.664 |
| audio_scan | 9.937 |
| asr_timings | 8.422 |
| ast_timings | 24.661 |
| describe_scenes | 13.143 |
| summarize_scenes | 26.616 |
| synthesize_synopsis | 39.918 |
| make_embedding | 3.059 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.232 |
| branch_yolo_total | 9.767 |
| branch_audio_total | 43.028 |
