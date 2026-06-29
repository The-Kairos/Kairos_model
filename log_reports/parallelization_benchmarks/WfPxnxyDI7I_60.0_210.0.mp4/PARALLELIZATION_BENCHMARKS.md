# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 20:54:02 UTC | WfPxnxyDI7I_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 132.682 | 0.660 | 50.985 | 11.033 | 7.500 | 9.956 | 3.300 |

## 2026-06-25 20:54:02 UTC | WfPxnxyDI7I_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/WfPxnxyDI7I_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `132.682` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.660 |
| save_clips | - |
| sample_frames | 0.893 |
| caption_frames | 36.730 |
| sample_fps | 2.036 |
| detect_object_yolo | 8.183 |
| audio_scan | 13.897 |
| asr_timings | 10.231 |
| ast_timings | 26.848 |
| describe_scenes | 11.033 |
| summarize_scenes | 7.500 |
| synthesize_synopsis | 9.956 |
| make_embedding | 3.300 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.629 |
| branch_yolo_total | 10.224 |
| branch_audio_total | 50.985 |
