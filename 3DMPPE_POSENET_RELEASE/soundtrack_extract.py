from moviepy.editor import *
videoclip = VideoFileClip('C:\\Users\\lenovo\\Desktop\\dance_videos\\better_when_im_dancing.mp4')
audioclip = videoclip.audio
print(audioclip.duration)