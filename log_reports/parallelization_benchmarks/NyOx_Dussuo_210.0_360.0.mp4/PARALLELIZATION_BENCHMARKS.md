# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 11:13:25 UTC | NyOx_Dussuo_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 129.153 | 0.783 | 43.109 | 8.732 | 19.087 | 20.246 | 2.307 |

## 2026-06-25 11:13:25 UTC | NyOx_Dussuo_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/NyOx_Dussuo_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `129.153` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.783 |
| save_clips | - |
| sample_frames | 0.586 |
| caption_frames | 23.864 |
| sample_fps | 2.001 |
| detect_object_yolo | 7.038 |
| audio_scan | 14.102 |
| asr_timings | 12.953 |
| ast_timings | 16.044 |
| describe_scenes | 8.732 |
| summarize_scenes | 19.087 |
| synthesize_synopsis | 20.246 |
| make_embedding | 2.307 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 24.457 |
| branch_yolo_total | 9.045 |
| branch_audio_total | 43.109 |
