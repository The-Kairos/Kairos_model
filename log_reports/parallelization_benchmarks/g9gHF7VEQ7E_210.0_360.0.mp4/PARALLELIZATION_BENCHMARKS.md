# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 05:00:56 UTC | g9gHF7VEQ7E_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 186.693 | 0.673 | 62.748 | 15.913 | 14.868 | 21.492 | 4.813 |

## 2026-06-26 05:00:56 UTC | g9gHF7VEQ7E_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/g9gHF7VEQ7E_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `186.693` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.673 |
| save_clips | - |
| sample_frames | 1.312 |
| caption_frames | 50.866 |
| sample_fps | 2.348 |
| detect_object_yolo | 10.222 |
| audio_scan | 14.112 |
| asr_timings | 9.611 |
| ast_timings | 39.016 |
| describe_scenes | 15.913 |
| summarize_scenes | 14.868 |
| synthesize_synopsis | 21.492 |
| make_embedding | 4.813 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 52.184 |
| branch_yolo_total | 12.576 |
| branch_audio_total | 62.748 |
