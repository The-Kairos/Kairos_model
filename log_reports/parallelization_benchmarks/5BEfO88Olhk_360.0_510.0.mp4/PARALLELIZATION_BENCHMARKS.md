# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 17:33:26 UTC | 5BEfO88Olhk_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 185.238 | 0.788 | 60.912 | 21.658 | 18.442 | 13.710 | 4.471 |
| 2026-06-24 11:26:23 UTC | 5BEfO88Olhk_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 185.403 | 0.798 | 60.647 | 22.533 | 15.932 | 14.890 | 4.444 |

## 2026-06-23 17:33:26 UTC | 5BEfO88Olhk_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/5BEfO88Olhk_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `185.238` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.788 |
| save_clips | - |
| sample_frames | 1.506 |
| caption_frames | 49.645 |
| sample_fps | 2.470 |
| detect_object_yolo | 10.266 |
| audio_scan | 10.554 |
| asr_timings | 12.483 |
| ast_timings | 37.867 |
| describe_scenes | 21.658 |
| summarize_scenes | 18.442 |
| synthesize_synopsis | 13.710 |
| make_embedding | 4.471 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.157 |
| branch_yolo_total | 12.742 |
| branch_audio_total | 60.912 |

## 2026-06-24 11:26:23 UTC | 5BEfO88Olhk_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/5BEfO88Olhk_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `185.403` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.798 |
| save_clips | - |
| sample_frames | 1.513 |
| caption_frames | 50.334 |
| sample_fps | 2.521 |
| detect_object_yolo | 10.394 |
| audio_scan | 10.670 |
| asr_timings | 12.172 |
| ast_timings | 37.796 |
| describe_scenes | 22.533 |
| summarize_scenes | 15.932 |
| synthesize_synopsis | 14.890 |
| make_embedding | 4.444 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.853 |
| branch_yolo_total | 12.921 |
| branch_audio_total | 60.647 |
