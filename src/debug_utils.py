def see_first_scene(scenes):
    if not scenes or not scenes[0]["frames"]:
        print("No frames to display.")
        return

    first_frame = scenes[0]["frames"][0]
    print(f"First scene frame info: {first_frame}")


def see_scenes_cuts(scenes):
    print("Detected scenes:")
    for i, s in enumerate(scenes):
        print(f"Scene {i}: start={s['start']} end={s['end']} frames={len(s['frames'])}")
