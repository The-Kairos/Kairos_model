# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 00:13:47 UTC | ExOC3jFZKAo_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 222.268 | 0.843 | 73.810 | 22.393 | 13.160 | 13.232 | 7.573 |

## 2026-06-25 00:13:47 UTC | ExOC3jFZKAo_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ExOC3jFZKAo_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `222.268` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.843 |
| save_clips | - |
| sample_frames | 1.944 |
| caption_frames | 71.847 |
| sample_fps | 2.852 |
| detect_object_yolo | 13.175 |
| audio_scan | 8.636 |
| asr_timings | 11.231 |
| ast_timings | 53.935 |
| describe_scenes | 22.393 |
| summarize_scenes | 13.160 |
| synthesize_synopsis | 13.232 |
| make_embedding | 7.573 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 73.797 |
| branch_yolo_total | 16.033 |
| branch_audio_total | 73.810 |
