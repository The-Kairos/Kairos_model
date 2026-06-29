# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 06:53:19 UTC | Kg38A1ggYiE_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 255.509 | 0.655 | 67.852 | 38.693 | 40.744 | 20.150 | 5.783 |

## 2026-06-25 06:53:19 UTC | Kg38A1ggYiE_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Kg38A1ggYiE_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `255.509` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.655 |
| save_clips | - |
| sample_frames | 1.360 |
| caption_frames | 64.248 |
| sample_fps | 2.350 |
| detect_object_yolo | 12.192 |
| audio_scan | 10.823 |
| asr_timings | 10.940 |
| ast_timings | 46.081 |
| describe_scenes | 38.693 |
| summarize_scenes | 40.744 |
| synthesize_synopsis | 20.150 |
| make_embedding | 5.783 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 65.614 |
| branch_yolo_total | 14.548 |
| branch_audio_total | 67.852 |
