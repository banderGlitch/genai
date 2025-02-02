import base64
from threading import Lock, Thread
import time
import numpy 
import cv2
import openai
from PIL import ImageGrab
from cv2 import imencode
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError

load_dotenv()

class ScreenCapture:
    def __init__(self):
        self.screenshot = None
        self.running = False
        self.lock = Lock()
        self.last_capture_time = time.time()
        self.capture_interval = 1/30  # 30 FPS target
        
    def start(self):
        if self.running:
            return self
        
        self.running = True
        self.thread = Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.running:
            current_time = time.time()
            if current_time - self.last_capture_time >= self.capture_interval:
                screenshot = ImageGrab.grab(bbox=None)
                screenshot = cv2.cvtColor(numpy.array(screenshot), cv2.COLOR_RGB2BGR)
                
                with self.lock:
                    self.screenshot = screenshot
                self.last_capture_time = current_time
            else:
                time.sleep(0.001)  # Small sleep to prevent CPU overload

    def read(self, encode=False):
        with self.lock:
            if self.screenshot is None:
                return None
            screenshot = self.screenshot.copy()

        if encode:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, buffer = imencode(".jpg", screenshot, encode_param)
            return base64.b64encode(buffer)

        return screenshot

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

class Assistant:
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)
        self.processing_lock = Lock()
        self.is_processing = False

    def answer(self, prompt, image):
        if not prompt or self.is_processing:
            return

        with self.processing_lock:
            self.is_processing = True
            try:
                print("Prompt:", prompt)
                response = self.chain.invoke(
                    {"prompt": prompt, "image_base64": image.decode()},
                    config={"configurable": {"session_id": "unused"}},
                ).strip()
                print("Response:", response)
                if response:
                    self._tts(response)
            finally:
                self.is_processing = False

    def _tts(self, response):
        player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)

        with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="shimmer",
            response_format="pcm",
            input=response,
        ) as stream:
            for chunk in stream.iter_bytes(chunk_size=1024):
                player.write(chunk)

    def _create_inference_chain(self, model):
        SYSTEM_PROMPT = """
        You are a witty assistant that will use the chat history and the image 
        provided by the user to answer its questions.

        Use few words on your answers. Go straight to the point. Do not use any
        emoticons or emojis. Do not ask the user any questions.

        Be friendly and helpful. Show some personality. Do not be too formal.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )

        chain = prompt_template | model | StrOutputParser()

        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )

# Initialize the screen capture
screen = ScreenCapture().start()

# Initialize the model and assistant
model = ChatOpenAI(model="gpt-4o", max_tokens=500)
assistant = Assistant(model)


# Set up speech recognition
def audio_callback(recognizer, audio):
    try:
        print("Processing audio...")
        prompt = recognizer.recognize_whisper(audio, model="tiny", language="english")
        if prompt:
            print(f"Detected: {prompt}")
            assistant.answer(prompt, screen.read(encode=True))
    except UnknownValueError:
        print("Could not understand audio")
    except Exception as e:
        print(f"Error processing audio: {str(e)}")

# Initialize speech recognition with optimized settings
recognizer = Recognizer()
recognizer.energy_threshold = 1000  # Adjust sensitivity
recognizer.dynamic_energy_threshold = False  # Disable dynamic adjustment
recognizer.pause_threshold = 0.5  # Reduce pause time
microphone = Microphone()

with microphone as source:
    recognizer.adjust_for_ambient_noise(source, duration=1)

stop_listening = recognizer.listen_in_background(
    microphone, 
    audio_callback,
    phrase_time_limit=5  # Limit max phrase length to 5 seconds
)

# Main loop
try:
    last_frame_time = time.time()
    fps = 0
    while True:
        frame = screen.read()
        if frame is not None:
            current_time = time.time()
            fps = 1 / (current_time - last_frame_time)
            last_frame_time = current_time
            
            # Only resize if we need to display
            if cv2.getWindowProperty("Screen Capture", cv2.WND_PROP_VISIBLE) > 0:
                scale_percent = 50
                width = int(frame.shape[1] * scale_percent / 100)
                height = int(frame.shape[0] * scale_percent / 100)
                resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                cv2.putText(resized, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Screen Capture", resized)
        
        if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
            break
        time.sleep(0.001)  # Prevent CPU overload

finally:
    # Cleanup
    screen.stop()
    cv2.destroyAllWindows()
    stop_listening(wait_for_stop=False)
    
# import base64
# from threading import Lock, Thread
# import time
# import numpy 
# import cv2
# import openai
# from PIL import ImageGrab
# from cv2 import imencode
# from dotenv import load_dotenv
# from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.schema.messages import SystemMessage
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_openai import ChatOpenAI
# from pyaudio import PyAudio, paInt16
# from speech_recognition import Microphone, Recognizer, UnknownValueError

# load_dotenv()

# class DesktopScreenshot:
#     def __init__(self):
#         self.screenshot = None
#         self.running = False
#         self.lock = Lock()

#     def start(self):
#         if self.running:
#             return self

#         self.running = True
#         self.thread = Thread(target=self.update, args=())
#         self.thread.start()
#         return self

#     def update(self):
#         while self.running:
#             screenshot = ImageGrab.grab()
#             screenshot = cv2.cvtColor(numpy.array(screenshot), cv2.COLOR_RGB2BGR)

#             self.lock.acquire()
#             self.screenshot = screenshot
#             self.lock.release()

#             time.sleep(0.1) 

#     def read(self, encode=False):
#         self.lock.acquire()
#         screenshot = self.screenshot.copy() if self.screenshot is not None else None
#         self.lock.release()

#         if encode and screenshot is not None:
#             _, buffer = imencode(".jpeg", screenshot)
#             return base64.b64encode(buffer)

#         return screenshot

#     def stop(self):
#         self.running = False
#         if self.thread.is_alive():
#             self.thread.join()


# class Assistant:
#     def __init__(self, model):
#         self.chain = self._create_inference_chain(model)

#     def answer(self, prompt, image):
#         if not prompt:
#             return

#         print("Prompt:", prompt)

#         response = self.chain.invoke(
#             {"prompt": prompt, "image_base64": image.decode()},
#             config={"configurable": {"session_id": "unused"}},
#         ).strip()

#         print("Response:", response)

#         if response:
#             self._tts(response)

#     def _tts(self, response):
#         player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)

#         with openai.audio.speech.with_streaming_response.create(
#             model="tts-1",
#             voice="shimmer",
#             response_format="pcm",
#             input=response,
#         ) as stream:
#             for chunk in stream.iter_bytes(chunk_size=1024):
#                 player.write(chunk)

#     def _create_inference_chain(self, model):
#         SYSTEM_PROMPT = """
#         You are a witty assistant that will use the chat history and the image 
#         provided by the user to answer its questions.

#         Use few words on your answers. Go straight to the point. Do not use any
#         emoticons or emojis. Do not ask the user any questions.

#         Be friendly and helpful. Show some personality. Do not be too formal.
#         """

#         prompt_template = ChatPromptTemplate.from_messages(
#             [
#                 SystemMessage(content=SYSTEM_PROMPT),
#                 MessagesPlaceholder(variable_name="chat_history"),
#                 (
#                     "human",
#                     [
#                         {"type": "text", "text": "{prompt}"},
#                         {
#                             "type": "image_url",
#                             "image_url": "data:image/jpeg;base64,{image_base64}",
#                         },
#                     ],
#                 ),
#             ]
#         )

#         chain = prompt_template | model | StrOutputParser()

#         chat_message_history = ChatMessageHistory()
#         return RunnableWithMessageHistory(
#             chain,
#             lambda _: chat_message_history,
#             input_messages_key="prompt",
#             history_messages_key="chat_history",
#         )


# desktop_screenshot = DesktopScreenshot().start()

# model = ChatOpenAI(model="gpt-4o")

# assistant = Assistant(model)


# def audio_callback(recognizer, audio):
#     try:
#         prompt = recognizer.recognize_whisper(audio, model="base", language="english")
#         assistant.answer(prompt, desktop_screenshot.read(encode=True))
#     except UnknownValueError:
#         print("There was an error processing the audio.")


# recognizer = Recognizer()
# microphone = Microphone()
# with microphone as source:
#     recognizer.adjust_for_ambient_noise(source)

# stop_listening = recognizer.listen_in_background(microphone, audio_callback)

# while True:
#     screenshot = desktop_screenshot.read()
#     if screenshot is not None:
#         cv2.imshow("Desktop", screenshot)
#     if cv2.waitKey(1) in [27, ord("q")]:
#         break

# desktop_screenshot.stop()
# cv2.destroyAllWindows()
# stop_listening(wait_for_stop=False)


