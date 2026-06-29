# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 10:31:33 UTC | N2ab7XwTNpg_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 304.806 | 0.704 | 73.872 | 18.584 | 102.941 | 16.292 | 6.315 |

## 2026-06-25 10:31:33 UTC | N2ab7XwTNpg_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/N2ab7XwTNpg_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `304.806` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.704 |
| save_clips | - |
| sample_frames | 1.942 |
| caption_frames | 67.628 |
| sample_fps | 2.691 |
| detect_object_yolo | 12.416 |
| audio_scan | 14.835 |
| asr_timings | 9.937 |
| ast_timings | 49.092 |
| describe_scenes | 18.584 |
| summarize_scenes | 102.941 |
| synthesize_synopsis | 16.292 |
| make_embedding | 6.315 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 69.576 |
| branch_yolo_total | 15.114 |
| branch_audio_total | 73.872 |
