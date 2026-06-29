# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 02:25:10 UTC | vbqZqioIK1c_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 80.456 | 0.612 | 32.756 | 3.869 | 5.410 | 11.280 | 1.799 |

## 2026-06-27 02:25:10 UTC | vbqZqioIK1c_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/vbqZqioIK1c_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `80.456` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.612 |
| save_clips | - |
| sample_frames | 0.310 |
| caption_frames | 14.835 |
| sample_fps | 1.704 |
| detect_object_yolo | 6.496 |
| audio_scan | 10.824 |
| asr_timings | 11.712 |
| ast_timings | 10.211 |
| describe_scenes | 3.869 |
| summarize_scenes | 5.410 |
| synthesize_synopsis | 11.280 |
| make_embedding | 1.799 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 15.151 |
| branch_yolo_total | 8.206 |
| branch_audio_total | 32.756 |
