from io import BytesIO
from langchain_community.llms import Ollama
from langchain_core.prompts.prompt import PromptTemplate
from picamera2 import Picamera2
from playsound import playsound
from pvrecorder import PvRecorder
from vertexai.preview.generative_models import GenerativeModel, Part, Image

import base64
import os
import pvporcupine
import speech_recognition as sr
import subprocess
import time
import netifaces
import sys


from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import vertexai.preview.generative_models as generative_models
import vertexai
import vertexai.preview.generative_models as generative_models

home_dir = "/home/anas/Documents/sight-beyond-seeing/"
audio_dir = home_dir + "audio/"
image_dir = home_dir + "images/"

picam2 = Picamera2()
picam2.options['quality'] = 80  # values from 0 to 100
r = sr.Recognizer()
r.energy_threshold = 300

porcupine = pvporcupine.create(access_key=os.environ["PORCUPINE_ACCESS_KEY"], keyword_paths=["/home/anas/Documents/sight-beyond-seeing/audio/Hey-Pixie_en_raspberry-pi_v3_0_0.ppn"])
devices = PvRecorder.get_available_devices()
recoder = PvRecorder(device_index=0, frame_length=porcupine.frame_length)

BYE = "bye.mp3"
HI = "hi.mp3"
HMIHU = "how_may_i_help_you.mp3"
IMREADY = "i_am_ready.mp3"
OK = "ok.mp3"
PLEASE_WAIT = "please_wait.mp3"
DING = "ding.mp3"
SURROUNDINGS = "surroundings.mp3"
SMS = "sms.mp3"
PICK_UP = "pick_up.mp3"
WORKING = "working.mp3"
DONE = "done.mp3"

top_k = 32
top_p = 0.5
temp = 0.3

active = False
image = None
categories = []
description = ""

llm = ChatGoogleGenerativeAI(model="gemini-pro-vision", safety_settings={
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        })

prompt = PromptTemplate.from_template("""Below is an instruction that describes a task, paired with an input that
provides further context.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}""")

tinyllm = Ollama(model="tinyllama_trained")
chain = prompt | tinyllm

def get_ssid():
    return subprocess.check_output(["/sbin/iwgetid -r"], shell = True).decode().rstrip()

def get_ip_add():
    import netifaces

def get_ip_address(interface):
    try:
        addrs = netifaces.ifaddresses(interface)
        ip_info = {}
        if netifaces.AF_INET in addrs:  # Check for IPv4 address
           return addrs[netifaces.AF_INET][0]['addr']
        if not ip_info:
            return "No IPv4 or IPv6 address found."
        return ip_info
    except ValueError:
        return "Interface not found or doesn't have an IP address."

def connection_details():
    print("Connected to WiFi " + get_ssid());
    speak("Connected to WiFi " + get_ssid())
    print("IP Address " + get_ip_address("wlan0"));
    speak("IP Address " + get_ip_address("wlan0"))

def load_image_from_disk(image_path: str) -> Image:
    f = open(image_path, mode="rb")
    image = Image.from_bytes(f.read())
    f.close()
    return image

def speak(text):
    if len(text.strip()) > 0:
        from gtts import gTTS 
        filename = audio_dir + "sight_beyond_seeing.mp3"
        language = 'en'
        myobj = gTTS(text=text, lang=language, slow=False) 
        myobj.save(filename)  
        os.system("mpg321 " + filename) 

def play_audio(filename, block=True):
    playsound(audio_dir + filename, block=block)

def capture_image():
    picam2.start()
    time.sleep(2)
    capture_config = picam2.create_still_configuration()
    image = picam2.switch_mode_and_capture_image(capture_config)
    return image

def capture_and_save_image():
    image = capture_image()
    file_name = write_image(image)
    #return file_name
    return image

def write_image(image):
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    filename = image_dir + timestamp + ".jpg"
    buff = BytesIO()
    image.save(buff, format="JPEG")
    with open(filename, "wb") as binary_file:
        binary_file.write(buff.getvalue())
    return filename                   

def prepare_image(image):
    buff = BytesIO()
    image.save(buff, format="JPEG")
    image_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    image1 = Part.from_data(data=image_b64,mime_type="image/jpeg")
    return image1

def generate(image, message):
    print("Sending remote request")
    prompts = [prepare_image(image), message]
    global top_k, top_p, temp
    vertexai.init(project="celestial-brand-415021", location="us-central1")
    model = GenerativeModel("gemini-pro-vision")
    buff = BytesIO()
    responses = model.generate_content(
       prompts,
        generation_config={
            "max_output_tokens": 1024,
            "temperature":temp,
            "top_p": top_p,
            "top_k": top_k
        },
        safety_settings={
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        },
        stream=False,
    )
    res = ""  
    print("Received response")
    for candidate in responses.candidates:
        for part in candidate.content.parts:
            res = res + part.text
    print("Response : ", res)
    return res

def generate_old(image, message):
    print("Sending remote request")
    global top_k, top_p, temp
    hmessage = HumanMessage(
        content=[
            {
                "type": "text",
                "text": message,
            },
            {"type": "image_url", "image_url": image},
            ]
        )
    message = llm.invoke([hmessage])
    res = message.content
    print(res)
    return res
def initialize(image):
    prompt = 'First, categorize the scene in one of the following categories ["Kitchen", " Living Room", " Bedroom", " Bathroom", " Sidewalk", " Street", " Park", " Bus Stop", " Train Station", ", Grocery Store", " Mall", " Restaurant", " Cafe", " Office", " Workspace", " Gym", " Theater", "Beach", " Forest", " Hiking Trail", " Classroom", " Library", " Hospital", " Clinic", " Exhibition", "Other"]. Then, starting on the following line, explain the scene. Use I see instead of using words, images, or scenes. Limit your response to a maximum of two sentences.'
    res = generate(image, prompt)
    lines = res.splitlines()
    if len(lines) > 0:
        objs = lines[0].strip().replace("Objects : ","").strip().split(",")
        if len(lines) > 1:
            scene = lines[-1].strip().replace("Scene :","").strip()
            speak(scene)
        return objs, scene 
    else:
        return "", "Sorry I could not find anything."

def process_txt(text):
    global image
    global active
    print(text, text.lower())
    if text.lower() == "wi-fi":
        connection_details()
    elif text.lower() == "bye":
        play_audio(DONE)
        play_audio(BYE)
        active = False
    elif len(text) > 0:
        play_audio(WORKING, False)
        rephrased = chain.invoke({"instruction" : "Rephrase the question", "input": text, "response" : ""})
        print("Rephrased Prompt : " + rephrased)
        res = generate(image, rephrased +  ". Answer in context of image")
        print(res)
        lines = res.split(".")
        for line in lines:
            print(line)
            speak(line)

def main():
    global image
    global active
    global categories
    global description

    play_audio(IMREADY)
    recoder.start()
    while True:
        audio = recoder.read()
        keyword_index = porcupine.process(audio)
        if keyword_index >= 0:
            print(f"Wake word Detected")
            play_audio(PICK_UP)
            image = capture_and_save_image()        
            play_audio(WORKING, False)
            categories, description = initialize(image)
            play_audio(HMIHU)
            active = True
        if active:
            with sr.Microphone() as source:
                print("Say something...")
                r.pause_threshold = 1
                audio = r.listen(source, 0, 6)
                try:
                    text = r.recognize_google(audio)
                    print(text)
                    if text.startswith("pixi"):
                        play_audio(PICK_UP)
                        print(f"Prompt Detected")
                        text = text.removeprefix('pixie').strip()
                        process_txt(text)
                except sr.UnknownValueError:
                    print(sr.UnknownValueError, file=sys.stderr)
                    print("Could not understand audio", file=sys.stderr)
                except sr.RequestError as e:
                    print("Could not request results from Speech Recognition service; {0}".format(e))
                print(active)
        
if __name__ == "__main__":
    main()
