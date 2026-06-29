# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 18:08:38 UTC | 9_UC5c7LFN4_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 139.168 | 0.820 | 40.415 | 14.442 | 14.331 | 30.865 | 2.618 |

## 2026-06-24 18:08:38 UTC | 9_UC5c7LFN4_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/9_UC5c7LFN4_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `139.168` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.820 |
| save_clips | - |
| sample_frames | 0.622 |
| caption_frames | 24.832 |
| sample_fps | 2.055 |
| detect_object_yolo | 6.738 |
| audio_scan | 8.660 |
| asr_timings | 13.066 |
| ast_timings | 18.681 |
| describe_scenes | 14.442 |
| summarize_scenes | 14.331 |
| synthesize_synopsis | 30.865 |
| make_embedding | 2.618 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.460 |
| branch_yolo_total | 8.799 |
| branch_audio_total | 40.415 |
