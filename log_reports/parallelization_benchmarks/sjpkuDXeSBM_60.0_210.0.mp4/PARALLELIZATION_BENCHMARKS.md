# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 20:16:01 UTC | sjpkuDXeSBM_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 208.658 | 0.830 | 88.620 | 17.919 | 11.157 | 17.718 | 4.650 |

## 2026-06-26 20:16:01 UTC | sjpkuDXeSBM_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/sjpkuDXeSBM_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `208.658` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.830 |
| save_clips | - |
| sample_frames | 1.531 |
| caption_frames | 51.994 |
| sample_fps | 2.421 |
| detect_object_yolo | 10.415 |
| audio_scan | 10.852 |
| asr_timings | 39.240 |
| ast_timings | 38.520 |
| describe_scenes | 17.919 |
| summarize_scenes | 11.157 |
| synthesize_synopsis | 17.718 |
| make_embedding | 4.650 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.531 |
| branch_yolo_total | 12.842 |
| branch_audio_total | 88.620 |
