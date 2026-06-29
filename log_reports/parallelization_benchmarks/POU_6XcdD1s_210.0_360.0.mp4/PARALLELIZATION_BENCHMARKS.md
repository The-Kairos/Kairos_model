# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 13:28:23 UTC | POU_6XcdD1s_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 159.884 | 0.631 | 72.829 | 12.153 | 15.522 | 20.870 | 2.660 |

## 2026-06-25 13:28:23 UTC | POU_6XcdD1s_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/POU_6XcdD1s_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `159.884` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.631 |
| save_clips | - |
| sample_frames | 0.768 |
| caption_frames | 24.278 |
| sample_fps | 1.933 |
| detect_object_yolo | 6.841 |
| audio_scan | 9.932 |
| asr_timings | 44.168 |
| ast_timings | 18.721 |
| describe_scenes | 12.153 |
| summarize_scenes | 15.522 |
| synthesize_synopsis | 20.870 |
| make_embedding | 2.660 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.052 |
| branch_yolo_total | 8.780 |
| branch_audio_total | 72.829 |
