# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 16:36:09 UTC | 3waJEHGNYug_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 165.064 | 0.628 | 58.444 | 17.376 | 16.172 | 13.536 | 3.908 |
| 2026-06-24 10:31:27 UTC | 3waJEHGNYug_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 177.230 | 0.665 | 57.659 | 21.327 | 11.106 | 27.775 | 3.930 |

## 2026-06-23 16:36:09 UTC | 3waJEHGNYug_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3waJEHGNYug_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `165.064` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.628 |
| save_clips | - |
| sample_frames | 1.011 |
| caption_frames | 41.561 |
| sample_fps | 2.113 |
| detect_object_yolo | 8.938 |
| audio_scan | 13.754 |
| asr_timings | 12.390 |
| ast_timings | 32.292 |
| describe_scenes | 17.376 |
| summarize_scenes | 16.172 |
| synthesize_synopsis | 13.536 |
| make_embedding | 3.908 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.579 |
| branch_yolo_total | 11.057 |
| branch_audio_total | 58.444 |

## 2026-06-24 10:31:27 UTC | 3waJEHGNYug_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3waJEHGNYug_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `177.230` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.665 |
| save_clips | - |
| sample_frames | 1.048 |
| caption_frames | 40.729 |
| sample_fps | 2.162 |
| detect_object_yolo | 9.362 |
| audio_scan | 13.884 |
| asr_timings | 11.245 |
| ast_timings | 32.522 |
| describe_scenes | 21.327 |
| summarize_scenes | 11.106 |
| synthesize_synopsis | 27.775 |
| make_embedding | 3.930 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.783 |
| branch_yolo_total | 11.531 |
| branch_audio_total | 57.659 |
