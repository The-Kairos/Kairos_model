# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 22:48:41 UTC | srP7RFVdjWc_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 310.195 | 0.832 | 112.897 | 34.394 | 21.485 | 6.160 | 7.629 |

## 2026-06-26 22:48:41 UTC | srP7RFVdjWc_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/srP7RFVdjWc_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `310.195` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.832 |
| save_clips | - |
| sample_frames | 2.732 |
| caption_frames | 110.159 |
| sample_fps | 3.325 |
| detect_object_yolo | 18.420 |
| audio_scan | 13.904 |
| asr_timings | 8.619 |
| ast_timings | 81.098 |
| describe_scenes | 34.394 |
| summarize_scenes | 21.485 |
| synthesize_synopsis | 6.160 |
| make_embedding | 7.629 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 112.897 |
| branch_yolo_total | 21.751 |
| branch_audio_total | 103.629 |
