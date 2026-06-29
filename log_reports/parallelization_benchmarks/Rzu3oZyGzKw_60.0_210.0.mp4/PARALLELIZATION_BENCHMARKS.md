# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 16:52:33 UTC | Rzu3oZyGzKw_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 175.165 | 0.766 | 79.005 | 13.664 | 13.727 | 20.566 | 3.817 |

## 2026-06-25 16:52:33 UTC | Rzu3oZyGzKw_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Rzu3oZyGzKw_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `175.165` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.766 |
| save_clips | - |
| sample_frames | 1.021 |
| caption_frames | 26.099 |
| sample_fps | 2.082 |
| detect_object_yolo | 8.430 |
| audio_scan | 16.446 |
| asr_timings | 32.878 |
| ast_timings | 29.673 |
| describe_scenes | 13.664 |
| summarize_scenes | 13.727 |
| synthesize_synopsis | 20.566 |
| make_embedding | 3.817 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.126 |
| branch_yolo_total | 10.518 |
| branch_audio_total | 79.005 |
