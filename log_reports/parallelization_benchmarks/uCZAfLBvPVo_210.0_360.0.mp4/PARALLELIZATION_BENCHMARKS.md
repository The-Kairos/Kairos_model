# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 00:31:44 UTC | uCZAfLBvPVo_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 112.315 | 0.690 | 38.106 | 11.377 | 6.932 | 7.196 | 3.344 |

## 2026-06-27 00:31:44 UTC | uCZAfLBvPVo_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/uCZAfLBvPVo_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `112.315` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.690 |
| save_clips | - |
| sample_frames | 1.240 |
| caption_frames | 36.860 |
| sample_fps | 2.168 |
| detect_object_yolo | 8.924 |
| audio_scan | 3.873 |
| asr_timings | 0.000 |
| ast_timings | 28.242 |
| describe_scenes | 11.377 |
| summarize_scenes | 6.932 |
| synthesize_synopsis | 7.196 |
| make_embedding | 3.344 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.106 |
| branch_yolo_total | 11.099 |
| branch_audio_total | 32.123 |
