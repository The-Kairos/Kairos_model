# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 00:33:38 UTC | Fptgkh2-2DQ_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 151.147 | 0.812 | 60.839 | 9.775 | 14.236 | 9.322 | 3.580 |

## 2026-06-25 00:33:38 UTC | Fptgkh2-2DQ_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Fptgkh2-2DQ_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `151.147` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.812 |
| save_clips | - |
| sample_frames | 1.246 |
| caption_frames | 38.664 |
| sample_fps | 2.262 |
| detect_object_yolo | 8.979 |
| audio_scan | 13.911 |
| asr_timings | 17.650 |
| ast_timings | 29.269 |
| describe_scenes | 9.775 |
| summarize_scenes | 14.236 |
| synthesize_synopsis | 9.322 |
| make_embedding | 3.580 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.916 |
| branch_yolo_total | 11.247 |
| branch_audio_total | 60.839 |
