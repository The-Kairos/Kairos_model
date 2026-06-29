# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 05:56:09 UTC | guxM1xmo9Gg_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 198.841 | 0.827 | 65.409 | 20.415 | 15.548 | 18.851 | 5.066 |

## 2026-06-26 05:56:09 UTC | guxM1xmo9Gg_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/guxM1xmo9Gg_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `198.841` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.827 |
| save_clips | - |
| sample_frames | 1.596 |
| caption_frames | 55.891 |
| sample_fps | 2.548 |
| detect_object_yolo | 11.181 |
| audio_scan | 13.138 |
| asr_timings | 11.722 |
| ast_timings | 40.540 |
| describe_scenes | 20.415 |
| summarize_scenes | 15.548 |
| synthesize_synopsis | 18.851 |
| make_embedding | 5.066 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 57.493 |
| branch_yolo_total | 13.735 |
| branch_audio_total | 65.409 |
