# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-11 07:38:35 UTC | 2025-08-02_13-15-11.mp4 | parallel | gemini | gemini-embedding-001 | 78.799 | 6.081 | 41.795 | 9.817 | 7.932 | 7.497 | 1.101 |

## 2026-04-11 07:38:35 UTC | 2025-08-02_13-15-11.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/412e0b18-e119-41d1-9317-e034ef4e613c/2025-08-02_13-15-11.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `78.799` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 6.081 |
| save_clips | - |
| sample_frames | 15.245 |
| caption_frames | 9.788 |
| sample_fps | 37.659 |
| detect_object_yolo | 4.128 |
| audio_scan | 4.920 |
| asr_timings | 8.946 |
| ast_timings | 22.750 |
| describe_scenes | 9.817 |
| summarize_scenes | 7.932 |
| synthesize_synopsis | 7.497 |
| make_embedding | 1.101 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.041 |
| branch_yolo_total | 41.795 |
| branch_audio_total | 27.681 |
