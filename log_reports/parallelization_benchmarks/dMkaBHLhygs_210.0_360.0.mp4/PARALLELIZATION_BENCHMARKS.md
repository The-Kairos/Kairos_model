# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 02:54:21 UTC | dMkaBHLhygs_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 136.124 | 0.682 | 48.795 | 11.302 | 17.937 | 15.324 | 2.819 |

## 2026-06-26 02:54:21 UTC | dMkaBHLhygs_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/dMkaBHLhygs_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `136.124` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.682 |
| save_clips | - |
| sample_frames | 0.806 |
| caption_frames | 27.463 |
| sample_fps | 2.005 |
| detect_object_yolo | 7.599 |
| audio_scan | 16.223 |
| asr_timings | 11.227 |
| ast_timings | 21.337 |
| describe_scenes | 11.302 |
| summarize_scenes | 17.937 |
| synthesize_synopsis | 15.324 |
| make_embedding | 2.819 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 28.275 |
| branch_yolo_total | 9.610 |
| branch_audio_total | 48.795 |
