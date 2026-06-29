# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 17:50:05 UTC | TBMt3agq6_w_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 170.437 | 0.812 | 55.315 | 14.012 | 22.930 | 15.422 | 3.941 |

## 2026-06-25 17:50:05 UTC | TBMt3agq6_w_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/TBMt3agq6_w_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `170.437` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.812 |
| save_clips | - |
| sample_frames | 1.261 |
| caption_frames | 43.216 |
| sample_fps | 2.369 |
| detect_object_yolo | 9.700 |
| audio_scan | 10.879 |
| asr_timings | 11.264 |
| ast_timings | 33.163 |
| describe_scenes | 14.012 |
| summarize_scenes | 22.930 |
| synthesize_synopsis | 15.422 |
| make_embedding | 3.941 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.484 |
| branch_yolo_total | 12.075 |
| branch_audio_total | 55.315 |
