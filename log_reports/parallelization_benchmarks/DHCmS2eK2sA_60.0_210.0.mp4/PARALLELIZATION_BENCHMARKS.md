# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 22:25:56 UTC | DHCmS2eK2sA_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 194.065 | 0.806 | 67.286 | 17.997 | 23.753 | 9.081 | 5.096 |

## 2026-06-24 22:25:56 UTC | DHCmS2eK2sA_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/DHCmS2eK2sA_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `194.065` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.806 |
| save_clips | - |
| sample_frames | 1.285 |
| caption_frames | 54.277 |
| sample_fps | 2.445 |
| detect_object_yolo | 10.611 |
| audio_scan | 15.264 |
| asr_timings | 10.977 |
| ast_timings | 41.037 |
| describe_scenes | 17.997 |
| summarize_scenes | 23.753 |
| synthesize_synopsis | 9.081 |
| make_embedding | 5.096 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 55.568 |
| branch_yolo_total | 13.062 |
| branch_audio_total | 67.286 |
