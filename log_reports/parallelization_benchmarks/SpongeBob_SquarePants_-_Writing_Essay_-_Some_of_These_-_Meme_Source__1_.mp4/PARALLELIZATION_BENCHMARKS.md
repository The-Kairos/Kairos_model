# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-10 16:15:08 UTC | SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1_.mp4 | parallel | gemini | gemini-embedding-001 | 61.749 | 0.437 | 17.108 | 14.875 | 5.456 | 18.394 | 0.813 |
| 2026-04-11 15:27:52 UTC | SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1_.mp4 | parallel | gemini | gemini-embedding-001 | 44.522 | 0.453 | 19.542 | 8.208 | 4.300 | 6.571 | 0.752 |

## 2026-04-10 16:15:08 UTC | SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1_.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/9a59a065-2a3f-449d-a2eb-98068d7706c6/SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1_.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `61.749` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.437 |
| save_clips | - |
| sample_frames | 0.756 |
| caption_frames | 5.286 |
| sample_fps | 1.332 |
| detect_object_yolo | 2.722 |
| audio_scan | 8.688 |
| asr_timings | 3.880 |
| ast_timings | 8.412 |
| describe_scenes | 14.875 |
| summarize_scenes | 5.456 |
| synthesize_synopsis | 18.394 |
| make_embedding | 0.813 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 6.048 |
| branch_yolo_total | 4.062 |
| branch_audio_total | 17.108 |

## 2026-04-11 15:27:52 UTC | SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1_.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/b32010cf-13d7-430d-8336-64ec08eefa63/SpongeBob_SquarePants_-_Writing_Essay_-_Some_of_These_-_Meme_Source__1_.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `44.522` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.453 |
| save_clips | - |
| sample_frames | 0.763 |
| caption_frames | 5.571 |
| sample_fps | 1.330 |
| detect_object_yolo | 2.668 |
| audio_scan | 8.567 |
| asr_timings | 3.865 |
| ast_timings | 10.967 |
| describe_scenes | 8.208 |
| summarize_scenes | 4.300 |
| synthesize_synopsis | 6.571 |
| make_embedding | 0.752 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 6.341 |
| branch_yolo_total | 4.005 |
| branch_audio_total | 19.542 |
