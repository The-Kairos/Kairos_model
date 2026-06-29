# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 15:07:16 UTC | lL-8C12lsu0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 222.775 | 0.677 | 70.802 | 26.312 | 26.008 | 16.286 | 6.158 |

## 2026-06-26 15:07:16 UTC | lL-8C12lsu0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/lL-8C12lsu0_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `222.775` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.677 |
| save_clips | - |
| sample_frames | 1.222 |
| caption_frames | 59.552 |
| sample_fps | 2.312 |
| detect_object_yolo | 12.012 |
| audio_scan | 14.039 |
| asr_timings | 10.926 |
| ast_timings | 45.829 |
| describe_scenes | 26.312 |
| summarize_scenes | 26.008 |
| synthesize_synopsis | 16.286 |
| make_embedding | 6.158 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 60.779 |
| branch_yolo_total | 14.330 |
| branch_audio_total | 70.802 |
