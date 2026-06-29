# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 00:24:36 UTC | adpHRmWxqFU_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 140.375 | 0.657 | 49.530 | 11.394 | 15.982 | 11.628 | 3.271 |

## 2026-06-26 00:24:36 UTC | adpHRmWxqFU_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/adpHRmWxqFU_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `140.375` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.657 |
| save_clips | - |
| sample_frames | 0.838 |
| caption_frames | 35.501 |
| sample_fps | 2.037 |
| detect_object_yolo | 8.135 |
| audio_scan | 13.746 |
| asr_timings | 8.426 |
| ast_timings | 27.350 |
| describe_scenes | 11.394 |
| summarize_scenes | 15.982 |
| synthesize_synopsis | 11.628 |
| make_embedding | 3.271 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.345 |
| branch_yolo_total | 10.177 |
| branch_audio_total | 49.530 |
