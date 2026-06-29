# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 14:14:50 UTC | Pdrc545bIl4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 175.144 | 0.791 | 51.573 | 18.690 | 32.481 | 17.280 | 3.369 |

## 2026-06-25 14:14:50 UTC | Pdrc545bIl4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Pdrc545bIl4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `175.144` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.791 |
| save_clips | - |
| sample_frames | 1.065 |
| caption_frames | 37.121 |
| sample_fps | 2.240 |
| detect_object_yolo | 9.049 |
| audio_scan | 13.397 |
| asr_timings | 11.129 |
| ast_timings | 27.038 |
| describe_scenes | 18.690 |
| summarize_scenes | 32.481 |
| synthesize_synopsis | 17.280 |
| make_embedding | 3.369 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.192 |
| branch_yolo_total | 11.296 |
| branch_audio_total | 51.573 |
