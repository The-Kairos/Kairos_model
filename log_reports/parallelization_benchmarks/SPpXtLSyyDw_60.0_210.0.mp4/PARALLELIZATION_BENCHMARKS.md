# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 17:08:21 UTC | SPpXtLSyyDw_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 112.714 | 0.849 | 43.307 | 11.240 | 7.349 | 15.776 | 2.379 |

## 2026-06-25 17:08:21 UTC | SPpXtLSyyDw_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/SPpXtLSyyDw_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `112.714` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.849 |
| save_clips | - |
| sample_frames | 0.543 |
| caption_frames | 21.010 |
| sample_fps | 1.952 |
| detect_object_yolo | 6.898 |
| audio_scan | 12.742 |
| asr_timings | 15.220 |
| ast_timings | 15.335 |
| describe_scenes | 11.240 |
| summarize_scenes | 7.349 |
| synthesize_synopsis | 15.776 |
| make_embedding | 2.379 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 21.559 |
| branch_yolo_total | 8.856 |
| branch_audio_total | 43.307 |
