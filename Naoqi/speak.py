from naoqi import ALProxy
tts = ALProxy("ALTextToSpeech", "192.168.3.44", 9559)
tts.say("Hello, my name is Nao. Nice to meet you")