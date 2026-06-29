# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 09:58:50 UTC | izWkhAfNNQg_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 223.198 | 0.831 | 55.682 | 23.229 | 49.411 | 24.923 | 4.188 |

## 2026-06-26 09:58:50 UTC | izWkhAfNNQg_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/izWkhAfNNQg_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `223.198` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.831 |
| save_clips | - |
| sample_frames | 1.767 |
| caption_frames | 49.312 |
| sample_fps | 2.581 |
| detect_object_yolo | 9.838 |
| audio_scan | 10.768 |
| asr_timings | 9.387 |
| ast_timings | 35.519 |
| describe_scenes | 23.229 |
| summarize_scenes | 49.411 |
| synthesize_synopsis | 24.923 |
| make_embedding | 4.188 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.085 |
| branch_yolo_total | 12.426 |
| branch_audio_total | 55.682 |
