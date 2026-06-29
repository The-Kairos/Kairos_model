# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 20:34:54 UTC | WVfXEIyanKY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 181.837 | 0.774 | 58.669 | 12.498 | 32.672 | 12.073 | 4.096 |

## 2026-06-25 20:34:54 UTC | WVfXEIyanKY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/WVfXEIyanKY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `181.837` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.774 |
| save_clips | - |
| sample_frames | 1.166 |
| caption_frames | 46.444 |
| sample_fps | 2.273 |
| detect_object_yolo | 9.762 |
| audio_scan | 11.801 |
| asr_timings | 11.229 |
| ast_timings | 35.631 |
| describe_scenes | 12.498 |
| summarize_scenes | 32.672 |
| synthesize_synopsis | 12.073 |
| make_embedding | 4.096 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.616 |
| branch_yolo_total | 12.041 |
| branch_audio_total | 58.669 |
