# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 17:32:05 UTC | Sl0DKauc_oU_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 193.428 | 0.793 | 65.634 | 24.668 | 12.019 | 16.507 | 5.522 |

## 2026-06-25 17:32:05 UTC | Sl0DKauc_oU_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Sl0DKauc_oU_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `193.428` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.793 |
| save_clips | - |
| sample_frames | 1.545 |
| caption_frames | 52.105 |
| sample_fps | 2.466 |
| detect_object_yolo | 10.757 |
| audio_scan | 13.785 |
| asr_timings | 11.194 |
| ast_timings | 40.646 |
| describe_scenes | 24.668 |
| summarize_scenes | 12.019 |
| synthesize_synopsis | 16.507 |
| make_embedding | 5.522 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.656 |
| branch_yolo_total | 13.229 |
| branch_audio_total | 65.634 |
