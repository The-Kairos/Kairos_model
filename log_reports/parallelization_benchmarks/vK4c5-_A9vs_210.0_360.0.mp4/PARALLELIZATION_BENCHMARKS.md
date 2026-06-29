# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 02:12:53 UTC | vK4c5-_A9vs_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 195.475 | 0.785 | 71.876 | 14.664 | 15.181 | 9.774 | 5.361 |

## 2026-06-27 02:12:53 UTC | vK4c5-_A9vs_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/vK4c5-_A9vs_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `195.475` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.785 |
| save_clips | - |
| sample_frames | 1.542 |
| caption_frames | 60.831 |
| sample_fps | 2.587 |
| detect_object_yolo | 11.456 |
| audio_scan | 16.167 |
| asr_timings | 11.647 |
| ast_timings | 44.053 |
| describe_scenes | 14.664 |
| summarize_scenes | 15.181 |
| synthesize_synopsis | 9.774 |
| make_embedding | 5.361 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 62.378 |
| branch_yolo_total | 14.049 |
| branch_audio_total | 71.876 |
