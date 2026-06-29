# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 20:18:02 UTC | COXt_GfXa2M_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 177.235 | 0.664 | 49.846 | 13.423 | 21.745 | 35.211 | 3.584 |

## 2026-06-24 20:18:02 UTC | COXt_GfXa2M_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/COXt_GfXa2M_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `177.235` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.664 |
| save_clips | - |
| sample_frames | 1.176 |
| caption_frames | 39.136 |
| sample_fps | 2.143 |
| detect_object_yolo | 8.826 |
| audio_scan | 9.717 |
| asr_timings | 10.393 |
| ast_timings | 29.727 |
| describe_scenes | 13.423 |
| summarize_scenes | 21.745 |
| synthesize_synopsis | 35.211 |
| make_embedding | 3.584 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.318 |
| branch_yolo_total | 10.975 |
| branch_audio_total | 49.846 |
