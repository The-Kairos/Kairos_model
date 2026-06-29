# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 21:28:46 UTC | XDY_KawH6Fw_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 175.225 | 0.657 | 61.746 | 18.815 | 13.339 | 13.572 | 4.459 |

## 2026-06-25 21:28:46 UTC | XDY_KawH6Fw_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/XDY_KawH6Fw_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `175.225` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.657 |
| save_clips | - |
| sample_frames | 1.173 |
| caption_frames | 48.024 |
| sample_fps | 2.202 |
| detect_object_yolo | 9.838 |
| audio_scan | 15.050 |
| asr_timings | 9.628 |
| ast_timings | 37.060 |
| describe_scenes | 18.815 |
| summarize_scenes | 13.339 |
| synthesize_synopsis | 13.572 |
| make_embedding | 4.459 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.203 |
| branch_yolo_total | 12.046 |
| branch_audio_total | 61.746 |
