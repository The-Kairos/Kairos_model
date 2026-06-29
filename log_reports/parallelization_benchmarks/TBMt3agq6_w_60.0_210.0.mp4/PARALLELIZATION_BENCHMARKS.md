# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 17:53:24 UTC | TBMt3agq6_w_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 197.618 | 0.785 | 68.649 | 15.158 | 25.224 | 14.856 | 5.046 |

## 2026-06-25 17:53:24 UTC | TBMt3agq6_w_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/TBMt3agq6_w_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `197.618` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.785 |
| save_clips | - |
| sample_frames | 1.508 |
| caption_frames | 51.732 |
| sample_fps | 2.449 |
| detect_object_yolo | 10.767 |
| audio_scan | 16.168 |
| asr_timings | 11.779 |
| ast_timings | 40.693 |
| describe_scenes | 15.158 |
| summarize_scenes | 25.224 |
| synthesize_synopsis | 14.856 |
| make_embedding | 5.046 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.246 |
| branch_yolo_total | 13.222 |
| branch_audio_total | 68.649 |
