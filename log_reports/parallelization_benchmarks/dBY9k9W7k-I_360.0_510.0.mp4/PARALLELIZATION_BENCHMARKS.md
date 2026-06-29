# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 02:42:10 UTC | dBY9k9W7k-I_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 114.338 | 0.800 | 48.326 | 9.227 | 7.796 | 11.973 | 2.279 |

## 2026-06-26 02:42:10 UTC | dBY9k9W7k-I_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/dBY9k9W7k-I_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `114.338` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.800 |
| save_clips | - |
| sample_frames | 0.567 |
| caption_frames | 22.777 |
| sample_fps | 1.996 |
| detect_object_yolo | 7.206 |
| audio_scan | 13.019 |
| asr_timings | 19.091 |
| ast_timings | 16.207 |
| describe_scenes | 9.227 |
| summarize_scenes | 7.796 |
| synthesize_synopsis | 11.973 |
| make_embedding | 2.279 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.349 |
| branch_yolo_total | 9.207 |
| branch_audio_total | 48.326 |
