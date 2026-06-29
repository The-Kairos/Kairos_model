# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 20:07:40 UTC | VtlI5SY9oQw_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 73.484 | 0.640 | 26.613 | 6.564 | 5.776 | 12.205 | 1.532 |

## 2026-06-25 20:07:40 UTC | VtlI5SY9oQw_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/VtlI5SY9oQw_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `73.484` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.640 |
| save_clips | - |
| sample_frames | 0.179 |
| caption_frames | 10.537 |
| sample_fps | 1.639 |
| detect_object_yolo | 6.394 |
| audio_scan | 8.563 |
| asr_timings | 11.000 |
| ast_timings | 7.042 |
| describe_scenes | 6.564 |
| summarize_scenes | 5.776 |
| synthesize_synopsis | 12.205 |
| make_embedding | 1.532 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 10.722 |
| branch_yolo_total | 8.039 |
| branch_audio_total | 26.613 |
