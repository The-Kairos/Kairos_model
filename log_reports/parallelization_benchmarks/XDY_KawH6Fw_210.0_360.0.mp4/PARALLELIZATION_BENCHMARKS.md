# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 21:23:02 UTC | XDY_KawH6Fw_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 152.809 | 0.675 | 56.730 | 13.242 | 9.773 | 10.103 | 3.818 |

## 2026-06-25 21:23:02 UTC | XDY_KawH6Fw_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/XDY_KawH6Fw_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `152.809` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.675 |
| save_clips | - |
| sample_frames | 1.214 |
| caption_frames | 44.026 |
| sample_fps | 2.190 |
| detect_object_yolo | 9.569 |
| audio_scan | 13.927 |
| asr_timings | 9.570 |
| ast_timings | 33.225 |
| describe_scenes | 13.242 |
| summarize_scenes | 9.773 |
| synthesize_synopsis | 10.103 |
| make_embedding | 3.818 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.246 |
| branch_yolo_total | 11.765 |
| branch_audio_total | 56.730 |
