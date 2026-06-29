# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 02:57:26 UTC | wQoQU0blZgw_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 144.940 | 0.833 | 71.511 | 10.262 | 6.690 | 12.852 | 2.767 |

## 2026-06-27 02:57:26 UTC | wQoQU0blZgw_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/wQoQU0blZgw_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `144.940` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.833 |
| save_clips | - |
| sample_frames | 0.815 |
| caption_frames | 28.219 |
| sample_fps | 2.149 |
| detect_object_yolo | 7.343 |
| audio_scan | 16.356 |
| asr_timings | 33.864 |
| ast_timings | 21.281 |
| describe_scenes | 10.262 |
| summarize_scenes | 6.690 |
| synthesize_synopsis | 12.852 |
| make_embedding | 2.767 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.040 |
| branch_yolo_total | 9.498 |
| branch_audio_total | 71.511 |
