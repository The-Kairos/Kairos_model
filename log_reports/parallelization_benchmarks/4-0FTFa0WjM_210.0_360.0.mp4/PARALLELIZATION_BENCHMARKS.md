# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 22:30:33 UTC | 4-0FTFa0WjM_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 119.229 | 0.631 | 48.422 | 5.496 | 9.430 | 7.692 | 3.145 |

## 2026-06-21 22:30:33 UTC | 4-0FTFa0WjM_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4-0FTFa0WjM_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `119.229` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.631 |
| save_clips | - |
| sample_frames | 0.888 |
| caption_frames | 31.734 |
| sample_fps | 2.004 |
| detect_object_yolo | 8.401 |
| audio_scan | 14.845 |
| asr_timings | 9.968 |
| ast_timings | 23.601 |
| describe_scenes | 5.496 |
| summarize_scenes | 9.430 |
| synthesize_synopsis | 7.692 |
| make_embedding | 3.145 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.628 |
| branch_yolo_total | 10.411 |
| branch_audio_total | 48.422 |
