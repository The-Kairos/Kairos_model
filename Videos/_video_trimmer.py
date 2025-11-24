from moviepy.editor import VideoFileClip

def trim_video(input_path, start_time, end_time, output_path):
    """
    Trims a video from start_time to end_time and saves it to output_path.
    
    Parameters:
    - input_path: str, path to the input video file
    - start_time: str, start time in "MM:SS" format
    - end_time: str, end time in "MM:SS" format
    - output_path: str, path to save the trimmed video
    """
    # Convert "MM:SS" to seconds
    def mmss_to_seconds(mmss):
        hours, minutes = map(int, mmss.split(':'))
        return hours * 60 + minutes 

    start_sec = mmss_to_seconds(start_time)
    end_sec = mmss_to_seconds(end_time)

    # Load video
    clip = VideoFileClip(input_path).subclip(start_sec, end_sec)
    
    # Write output
    clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    clip.close()

# Example usage:
trim_video("Videos/UDST CCIT graduation.mp4", "42:13", "72:13", "Videos/UDST CCIT graduation 30 mins.mp4")
