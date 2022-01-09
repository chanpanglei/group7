from pygame import mixer
import time

mixer.init()
mixer.music.load('thx.mp3')
mixer.music.play()
time.sleep(8)
mixer.music.stop()
