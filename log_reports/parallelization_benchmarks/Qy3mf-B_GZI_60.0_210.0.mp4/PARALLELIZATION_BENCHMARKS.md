# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 16:14:23 UTC | Qy3mf-B_GZI_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 292.982 | 0.796 | 82.924 | 31.527 | 42.168 | 17.225 | 19.127 |

## 2026-06-25 16:14:23 UTC | Qy3mf-B_GZI_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Qy3mf-B_GZI_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `292.982` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.796 |
| save_clips | - |
| sample_frames | 1.723 |
| caption_frames | 81.194 |
| sample_fps | 2.774 |
| detect_object_yolo | 13.896 |
| audio_scan | 16.922 |
| asr_timings | 10.079 |
| ast_timings | 54.055 |
| describe_scenes | 31.527 |
| summarize_scenes | 42.168 |
| synthesize_synopsis | 17.225 |
| make_embedding | 19.127 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 82.924 |
| branch_yolo_total | 16.676 |
| branch_audio_total | 81.065 |
