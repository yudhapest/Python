import pyttsx3  # pip install pyttsx
import datetime
import speech_recognition as sr  # pip install SpeechRecognition

engine = pyttsx3.init()
engine.say("Hello My name is jarvis")
engine.runAndWait()


def speak(audio):
    engine.say(audio)
    engine.runAndWait()


def time_():
    Time = datetime.datetime.now().strftime("%I:%M:%S")
    speak("The current time is")
    speak(Time)


def date_():
    year = datetime.datetime.now().year
    month = datetime.datetime.now().month
    date = datetime.datetime.now().day
    speak("The current date is")
    speak(date)
    speak(month)
    speak(year)


def wisme():
    speak("Welcome back Brother!")
    time_()
    date_()

    # Greeting

    hour = datetime.datetime.now().hour

    if hour >= 6 and hour < 12:
        speak("Good Morning Brother!")
    if hour >= 12 and hour <= 18:
        speak("Good Afternoon Brother!")
    if hour >= 18 and hour < 24:
        speak("Good Night Brother!")

    speak("Jarvis at your service. Please tell me how can I help you today?")


def TakeCommand():
    SR = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listeing.....")
        SR.pause_threshold = 1
        audio = SR.listen(source)

    try:
        print("Recognizing.....")
        query = SR.recognize_google(audio, language='en-US')
        print(query)
    except Exception as e:
        print(e)
        print("Say that again please.....")
        return "None"
    return query


TakeCommand()
