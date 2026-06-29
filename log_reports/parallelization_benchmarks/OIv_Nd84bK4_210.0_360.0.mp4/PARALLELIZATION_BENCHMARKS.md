# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 11:27:54 UTC | OIv_Nd84bK4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 188.002 | 0.893 | 64.120 | 25.686 | 21.300 | 21.541 | 4.562 |

## 2026-06-25 11:27:54 UTC | OIv_Nd84bK4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/OIv_Nd84bK4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `188.002` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.893 |
| save_clips | - |
| sample_frames | 1.350 |
| caption_frames | 30.262 |
| sample_fps | 2.381 |
| detect_object_yolo | 9.359 |
| audio_scan | 17.522 |
| asr_timings | 9.159 |
| ast_timings | 37.431 |
| describe_scenes | 25.686 |
| summarize_scenes | 21.300 |
| synthesize_synopsis | 21.541 |
| make_embedding | 4.562 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.617 |
| branch_yolo_total | 11.745 |
| branch_audio_total | 64.120 |
