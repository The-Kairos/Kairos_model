# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 06:08:27 UTC | zeWShkauzL0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 151.138 | 0.752 | 55.985 | 14.370 | 9.523 | 4.950 | 4.207 |

## 2026-06-27 06:08:27 UTC | zeWShkauzL0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/zeWShkauzL0_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `151.138` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.752 |
| save_clips | - |
| sample_frames | 1.165 |
| caption_frames | 46.810 |
| sample_fps | 2.203 |
| detect_object_yolo | 9.778 |
| audio_scan | 11.734 |
| asr_timings | 8.165 |
| ast_timings | 36.078 |
| describe_scenes | 14.370 |
| summarize_scenes | 9.523 |
| synthesize_synopsis | 4.950 |
| make_embedding | 4.207 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.981 |
| branch_yolo_total | 11.986 |
| branch_audio_total | 55.985 |
