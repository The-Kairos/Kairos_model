# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 01:27:32 UTC | GErRlbPkMmQ_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 97.255 | 0.797 | 40.714 | 7.387 | 4.295 | 11.221 | 2.289 |

## 2026-06-25 01:27:32 UTC | GErRlbPkMmQ_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/GErRlbPkMmQ_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `97.255` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.797 |
| save_clips | - |
| sample_frames | 0.538 |
| caption_frames | 19.494 |
| sample_fps | 1.966 |
| detect_object_yolo | 7.173 |
| audio_scan | 15.902 |
| asr_timings | 9.608 |
| ast_timings | 15.195 |
| describe_scenes | 7.387 |
| summarize_scenes | 4.295 |
| synthesize_synopsis | 11.221 |
| make_embedding | 2.289 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 20.038 |
| branch_yolo_total | 9.145 |
| branch_audio_total | 40.714 |
