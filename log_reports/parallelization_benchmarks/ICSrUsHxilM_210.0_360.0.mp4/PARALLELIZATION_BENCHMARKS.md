# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 04:08:36 UTC | ICSrUsHxilM_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 153.041 | 0.783 | 58.061 | 15.118 | 11.283 | 10.652 | 3.637 |

## 2026-06-25 04:08:36 UTC | ICSrUsHxilM_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ICSrUsHxilM_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `153.041` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.783 |
| save_clips | - |
| sample_frames | 1.062 |
| caption_frames | 39.979 |
| sample_fps | 2.278 |
| detect_object_yolo | 8.783 |
| audio_scan | 15.952 |
| asr_timings | 11.784 |
| ast_timings | 30.316 |
| describe_scenes | 15.118 |
| summarize_scenes | 11.283 |
| synthesize_synopsis | 10.652 |
| make_embedding | 3.637 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.047 |
| branch_yolo_total | 11.067 |
| branch_audio_total | 58.061 |
