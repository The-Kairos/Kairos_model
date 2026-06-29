# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 01:50:27 UTC | GPZ4So5mepU_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 109.572 | 0.696 | 37.718 | 5.297 | 7.382 | 24.736 | 2.270 |

## 2026-06-25 01:50:27 UTC | GPZ4So5mepU_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/GPZ4So5mepU_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `109.572` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.696 |
| save_clips | - |
| sample_frames | 0.685 |
| caption_frames | 20.217 |
| sample_fps | 1.911 |
| detect_object_yolo | 7.214 |
| audio_scan | 12.831 |
| asr_timings | 9.086 |
| ast_timings | 15.793 |
| describe_scenes | 5.297 |
| summarize_scenes | 7.382 |
| synthesize_synopsis | 24.736 |
| make_embedding | 2.270 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 20.909 |
| branch_yolo_total | 9.132 |
| branch_audio_total | 37.718 |
