# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 03:16:09 UTC | wcXIN8aFfi4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 171.675 | 0.638 | 63.571 | 13.407 | 7.060 | 10.369 | 5.359 |

## 2026-06-27 03:16:09 UTC | wcXIN8aFfi4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/wcXIN8aFfi4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `171.675` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.638 |
| save_clips | - |
| sample_frames | 1.227 |
| caption_frames | 55.192 |
| sample_fps | 2.261 |
| detect_object_yolo | 11.182 |
| audio_scan | 11.913 |
| asr_timings | 7.429 |
| ast_timings | 44.221 |
| describe_scenes | 13.407 |
| summarize_scenes | 7.060 |
| synthesize_synopsis | 10.369 |
| make_embedding | 5.359 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 56.424 |
| branch_yolo_total | 13.449 |
| branch_audio_total | 63.571 |
