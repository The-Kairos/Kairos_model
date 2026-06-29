# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 14:14:38 UTC | kr_sTfYX_FI_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 130.071 | 0.808 | 46.617 | 13.349 | 14.647 | 16.255 | 2.590 |

## 2026-06-26 14:14:38 UTC | kr_sTfYX_FI_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/kr_sTfYX_FI_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `130.071` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.808 |
| save_clips | - |
| sample_frames | 0.645 |
| caption_frames | 24.561 |
| sample_fps | 2.051 |
| detect_object_yolo | 7.135 |
| audio_scan | 15.142 |
| asr_timings | 13.109 |
| ast_timings | 18.357 |
| describe_scenes | 13.349 |
| summarize_scenes | 14.647 |
| synthesize_synopsis | 16.255 |
| make_embedding | 2.590 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.212 |
| branch_yolo_total | 9.192 |
| branch_audio_total | 46.617 |
