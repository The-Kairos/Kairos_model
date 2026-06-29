# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 22:36:34 UTC | Za_exvdK2RQ_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 194.718 | 0.833 | 68.729 | 15.935 | 20.014 | 10.542 | 5.388 |

## 2026-06-25 22:36:34 UTC | Za_exvdK2RQ_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Za_exvdK2RQ_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `194.718` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.833 |
| save_clips | - |
| sample_frames | 1.660 |
| caption_frames | 56.501 |
| sample_fps | 2.601 |
| detect_object_yolo | 11.098 |
| audio_scan | 14.876 |
| asr_timings | 8.964 |
| ast_timings | 44.882 |
| describe_scenes | 15.935 |
| summarize_scenes | 20.014 |
| synthesize_synopsis | 10.542 |
| make_embedding | 5.388 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 58.166 |
| branch_yolo_total | 13.706 |
| branch_audio_total | 68.729 |
