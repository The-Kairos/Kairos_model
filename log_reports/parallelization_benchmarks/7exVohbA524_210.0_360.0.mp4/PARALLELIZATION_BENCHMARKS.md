# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 15:04:59 UTC | 7exVohbA524_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 2214.032 | 0.664 | 2083.535 | 27.036 | 16.646 | 17.031 | 4.498 |

## 2026-06-24 15:04:59 UTC | 7exVohbA524_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/7exVohbA524_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `2214.032` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.664 |
| save_clips | - |
| sample_frames | 1.272 |
| caption_frames | 49.586 |
| sample_fps | 2.251 |
| detect_object_yolo | 10.121 |
| audio_scan | 9.584 |
| asr_timings | 2035.671 |
| ast_timings | 38.272 |
| describe_scenes | 27.036 |
| summarize_scenes | 16.646 |
| synthesize_synopsis | 17.031 |
| make_embedding | 4.498 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.864 |
| branch_yolo_total | 12.378 |
| branch_audio_total | 2083.535 |
