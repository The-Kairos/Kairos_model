# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 12:11:20 UTC | 5W8Wc_u78VU_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 225.728 | 0.656 | 68.978 | 27.007 | 21.679 | 19.578 | 6.182 |

## 2026-06-24 12:11:20 UTC | 5W8Wc_u78VU_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/5W8Wc_u78VU_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `225.728` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.656 |
| save_clips | - |
| sample_frames | 1.581 |
| caption_frames | 64.221 |
| sample_fps | 2.541 |
| detect_object_yolo | 11.918 |
| audio_scan | 10.659 |
| asr_timings | 9.962 |
| ast_timings | 48.348 |
| describe_scenes | 27.007 |
| summarize_scenes | 21.679 |
| synthesize_synopsis | 19.578 |
| make_embedding | 6.182 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 65.808 |
| branch_yolo_total | 14.465 |
| branch_audio_total | 68.978 |
