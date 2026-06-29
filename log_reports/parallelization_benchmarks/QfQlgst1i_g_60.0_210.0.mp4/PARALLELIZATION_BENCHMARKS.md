# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 15:18:10 UTC | QfQlgst1i_g_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 214.479 | 0.775 | 64.405 | 32.067 | 27.008 | 20.242 | 4.505 |

## 2026-06-25 15:18:10 UTC | QfQlgst1i_g_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/QfQlgst1i_g_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `214.479` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.775 |
| save_clips | - |
| sample_frames | 1.138 |
| caption_frames | 50.153 |
| sample_fps | 2.373 |
| detect_object_yolo | 10.361 |
| audio_scan | 15.712 |
| asr_timings | 11.247 |
| ast_timings | 37.438 |
| describe_scenes | 32.067 |
| summarize_scenes | 27.008 |
| synthesize_synopsis | 20.242 |
| make_embedding | 4.505 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.297 |
| branch_yolo_total | 12.740 |
| branch_audio_total | 64.405 |
