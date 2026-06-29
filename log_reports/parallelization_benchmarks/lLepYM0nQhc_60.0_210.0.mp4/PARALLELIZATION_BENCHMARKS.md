# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 15:16:46 UTC | lLepYM0nQhc_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 226.240 | 0.795 | 75.746 | 26.587 | 27.989 | 14.342 | 5.452 |

## 2026-06-26 15:16:46 UTC | lLepYM0nQhc_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/lLepYM0nQhc_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `226.240` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.795 |
| save_clips | - |
| sample_frames | 1.488 |
| caption_frames | 57.935 |
| sample_fps | 2.601 |
| detect_object_yolo | 11.801 |
| audio_scan | 12.016 |
| asr_timings | 18.851 |
| ast_timings | 44.871 |
| describe_scenes | 26.587 |
| summarize_scenes | 27.989 |
| synthesize_synopsis | 14.342 |
| make_embedding | 5.452 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 59.429 |
| branch_yolo_total | 14.408 |
| branch_audio_total | 75.746 |
