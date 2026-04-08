# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-08 09:59:38 UTC | SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1___2__clip__7_.mp4 | parallel | gemini | gemini-embedding-001 | 35.871 | 0.153 | 6.785 | 8.051 | 6.033 | 10.174 | 0.712 |

## 2026-04-08 09:59:38 UTC | SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1___2__clip__7_.mp4 | parallel

- Video path: `/var/tmp/kairos/jobs/d3797752-1e07-4691-89c9-2b7d57bc2168/SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1___2__clip__7_.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `35.871` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.153 |
| save_clips | - |
| sample_frames | 0.099 |
| caption_frames | 2.209 |
| sample_fps | 0.157 |
| detect_object_yolo | 1.041 |
| audio_scan | 3.456 |
| asr_timings | 3.238 |
| ast_timings | 3.320 |
| describe_scenes | 8.051 |
| summarize_scenes | 6.033 |
| synthesize_synopsis | 10.174 |
| make_embedding | 0.712 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 2.315 |
| branch_yolo_total | 1.206 |
| branch_audio_total | 6.785 |
