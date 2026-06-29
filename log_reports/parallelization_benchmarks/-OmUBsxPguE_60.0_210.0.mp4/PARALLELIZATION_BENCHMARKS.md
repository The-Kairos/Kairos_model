# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 08:12:11 UTC | -OmUBsxPguE_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 193.304 | 0.679 | 55.796 | 27.089 | 13.094 | 28.827 | 4.526 |

## 2026-06-24 08:12:11 UTC | -OmUBsxPguE_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-OmUBsxPguE_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `193.304` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.679 |
| save_clips | - |
| sample_frames | 1.239 |
| caption_frames | 48.571 |
| sample_fps | 2.224 |
| detect_object_yolo | 9.894 |
| audio_scan | 6.463 |
| asr_timings | 11.499 |
| ast_timings | 37.826 |
| describe_scenes | 27.089 |
| summarize_scenes | 13.094 |
| synthesize_synopsis | 28.827 |
| make_embedding | 4.526 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.816 |
| branch_yolo_total | 12.125 |
| branch_audio_total | 55.796 |
