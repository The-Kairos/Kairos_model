# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 04:09:19 UTC | fMActfTJboI_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 124.218 | 0.804 | 46.339 | 8.312 | 11.128 | 12.893 | 2.763 |

## 2026-06-26 04:09:19 UTC | fMActfTJboI_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/fMActfTJboI_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `124.218` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.804 |
| save_clips | - |
| sample_frames | 0.913 |
| caption_frames | 29.046 |
| sample_fps | 2.190 |
| detect_object_yolo | 8.348 |
| audio_scan | 15.394 |
| asr_timings | 9.992 |
| ast_timings | 20.944 |
| describe_scenes | 8.312 |
| summarize_scenes | 11.128 |
| synthesize_synopsis | 12.893 |
| make_embedding | 2.763 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.965 |
| branch_yolo_total | 10.544 |
| branch_audio_total | 46.339 |
