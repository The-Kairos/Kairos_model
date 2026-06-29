# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 05:19:57 UTC | gehzWEPLjcc_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 161.458 | 0.717 | 50.138 | 14.110 | 21.394 | 21.035 | 3.325 |

## 2026-06-26 05:19:57 UTC | gehzWEPLjcc_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/gehzWEPLjcc_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `161.458` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.717 |
| save_clips | - |
| sample_frames | 1.258 |
| caption_frames | 37.166 |
| sample_fps | 2.268 |
| detect_object_yolo | 8.631 |
| audio_scan | 14.083 |
| asr_timings | 8.779 |
| ast_timings | 27.268 |
| describe_scenes | 14.110 |
| summarize_scenes | 21.394 |
| synthesize_synopsis | 21.035 |
| make_embedding | 3.325 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.429 |
| branch_yolo_total | 10.904 |
| branch_audio_total | 50.138 |
