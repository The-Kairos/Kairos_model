# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-10 18:00:07 UTC | Young_Sheldon.mp4 | parallel | gemini | gemini-embedding-001 | 151.476 | 1.620 | 73.460 | 46.486 | 10.929 | 10.932 | 2.871 |

## 2026-04-10 18:00:07 UTC | Young_Sheldon.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/f8ea68e8-b01d-4308-bf6c-aba8bba912fe/Young_Sheldon.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `151.476` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.620 |
| save_clips | - |
| sample_frames | 4.680 |
| caption_frames | 44.226 |
| sample_fps | 9.614 |
| detect_object_yolo | 15.502 |
| audio_scan | 27.186 |
| asr_timings | 11.426 |
| ast_timings | 46.262 |
| describe_scenes | 46.486 |
| summarize_scenes | 10.929 |
| synthesize_synopsis | 10.932 |
| make_embedding | 2.871 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.913 |
| branch_yolo_total | 25.124 |
| branch_audio_total | 73.460 |
