# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 10:48:09 UTC | NkmxlSJMYYs_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 103.922 | 0.816 | 38.533 | 5.229 | 4.887 | 28.326 | 1.854 |

## 2026-06-25 10:48:09 UTC | NkmxlSJMYYs_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/NkmxlSJMYYs_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `103.922` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.816 |
| save_clips | - |
| sample_frames | 0.341 |
| caption_frames | 14.564 |
| sample_fps | 1.843 |
| detect_object_yolo | 6.142 |
| audio_scan | 12.941 |
| asr_timings | 15.480 |
| ast_timings | 10.103 |
| describe_scenes | 5.229 |
| summarize_scenes | 4.887 |
| synthesize_synopsis | 28.326 |
| make_embedding | 1.854 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 14.911 |
| branch_yolo_total | 7.990 |
| branch_audio_total | 38.533 |
