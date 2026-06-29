# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 22:11:30 UTC | CyAyEewgdEc_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 1199.655 | 0.780 | 1119.819 | 13.358 | 9.313 | 8.229 | 3.064 |

## 2026-06-24 22:11:30 UTC | CyAyEewgdEc_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/CyAyEewgdEc_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `1199.655` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.780 |
| save_clips | - |
| sample_frames | 0.842 |
| caption_frames | 32.358 |
| sample_fps | 2.100 |
| detect_object_yolo | 8.375 |
| audio_scan | 10.772 |
| asr_timings | 1084.681 |
| ast_timings | 24.357 |
| describe_scenes | 13.358 |
| summarize_scenes | 9.313 |
| synthesize_synopsis | 8.229 |
| make_embedding | 3.064 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.205 |
| branch_yolo_total | 10.482 |
| branch_audio_total | 1119.819 |
