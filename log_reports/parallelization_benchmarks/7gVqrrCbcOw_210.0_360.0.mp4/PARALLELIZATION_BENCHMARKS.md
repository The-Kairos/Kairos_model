# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 16:27:59 UTC | 7gVqrrCbcOw_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 181.643 | 0.810 | 60.796 | 21.894 | 10.858 | 18.988 | 4.183 |

## 2026-06-24 16:27:59 UTC | 7gVqrrCbcOw_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/7gVqrrCbcOw_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `181.643` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.810 |
| save_clips | - |
| sample_frames | 1.195 |
| caption_frames | 49.286 |
| sample_fps | 2.347 |
| detect_object_yolo | 9.877 |
| audio_scan | 15.989 |
| asr_timings | 9.221 |
| ast_timings | 35.578 |
| describe_scenes | 21.894 |
| summarize_scenes | 10.858 |
| synthesize_synopsis | 18.988 |
| make_embedding | 4.183 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.487 |
| branch_yolo_total | 12.230 |
| branch_audio_total | 60.796 |
