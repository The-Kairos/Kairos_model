# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 12:49:11 UTC | -eQfg7HJuKc_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 182.718 | 0.793 | 68.643 | 17.565 | 11.864 | 9.275 | 5.104 |

## 2026-06-27 12:49:11 UTC | -eQfg7HJuKc_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-eQfg7HJuKc_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `182.718` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.793 |
| save_clips | - |
| sample_frames | 1.812 |
| caption_frames | 52.369 |
| sample_fps | 2.604 |
| detect_object_yolo | 11.287 |
| audio_scan | 13.905 |
| asr_timings | 13.520 |
| ast_timings | 41.210 |
| describe_scenes | 17.565 |
| summarize_scenes | 11.864 |
| synthesize_synopsis | 9.275 |
| make_embedding | 5.104 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.186 |
| branch_yolo_total | 13.896 |
| branch_audio_total | 68.643 |
