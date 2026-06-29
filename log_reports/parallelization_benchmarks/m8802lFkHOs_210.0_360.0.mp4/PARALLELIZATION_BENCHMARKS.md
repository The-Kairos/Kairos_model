# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 17:57:46 UTC | m8802lFkHOs_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 279.019 | 0.879 | 77.410 | 39.325 | 49.779 | 20.183 | 6.308 |

## 2026-06-26 17:57:46 UTC | m8802lFkHOs_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/m8802lFkHOs_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `279.019` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.879 |
| save_clips | - |
| sample_frames | 1.698 |
| caption_frames | 66.805 |
| sample_fps | 2.709 |
| detect_object_yolo | 12.498 |
| audio_scan | 15.045 |
| asr_timings | 10.488 |
| ast_timings | 51.869 |
| describe_scenes | 39.325 |
| summarize_scenes | 49.779 |
| synthesize_synopsis | 20.183 |
| make_embedding | 6.308 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 68.508 |
| branch_yolo_total | 15.214 |
| branch_audio_total | 77.410 |
