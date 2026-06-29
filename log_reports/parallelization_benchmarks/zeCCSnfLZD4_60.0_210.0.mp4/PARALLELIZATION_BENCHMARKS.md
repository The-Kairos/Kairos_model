# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 05:47:32 UTC | zeCCSnfLZD4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 91.980 | 0.776 | 35.369 | 4.816 | 6.043 | 9.157 | 2.269 |

## 2026-06-27 05:47:32 UTC | zeCCSnfLZD4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/zeCCSnfLZD4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `91.980` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.776 |
| save_clips | - |
| sample_frames | 0.394 |
| caption_frames | 22.631 |
| sample_fps | 1.917 |
| detect_object_yolo | 7.215 |
| audio_scan | 8.513 |
| asr_timings | 11.156 |
| ast_timings | 15.691 |
| describe_scenes | 4.816 |
| summarize_scenes | 6.043 |
| synthesize_synopsis | 9.157 |
| make_embedding | 2.269 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.031 |
| branch_yolo_total | 9.138 |
| branch_audio_total | 35.369 |
