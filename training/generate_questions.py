from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
import os
import re
import csv
from dotenv import load_dotenv
load_dotenv() 

top_k = 32
top_p = 0.5
temperature = 0.4
max_output_tokens = 8048

vision_llm = ChatGoogleGenerativeAI(model="gemini-pro-vision", top_k=top_k, top_p=top_p, temperature=temperature, max_output_tokens=max_output_tokens, safety_settings={
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH, HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE, HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
})
txt_llm = ChatGoogleGenerativeAI(model="gemini-pro", top_k=top_k, top_p=top_p,
                                 temperature=temperature, max_output_tokens=max_output_tokens)
# txt_llm = ChatGoogleGenerativeAI(model="text-bison", top_k = top_k, top_p=top_p,temperature=temperature, max_output_tokens=max_output_tokens)

image_path = "./training/images/self_taken/"


def load_image_from_disk(image_path: str) -> Image:
    f = open(image_path, mode="rb")
    image = Image.from_bytes(f.read())
    f.close()
    return image


def generate_description(image_name):
    message = """You are an expert with an eye for details. Describe the image in as much as possible details."""
    hmessage = HumanMessage(
        content=[
            {
                "type": "text",
                "text": message,
            },
            {"type": "image_url", "image_url": image_name},
        ]
    )
    message = vision_llm.invoke([hmessage])
    res = message.content
    return res


def generate_questions(image_name):
    message = """You are an expert with an eye for details. Create a set of 30 questions which can be answered about this image to provide details to a blind person. Limit your response to the information available in the image and provide list of questions."""
    hmessage = HumanMessage(
        content=[
            {
                "type": "text",
                "text": message,
            },
            {"type": "image_url", "image_url": image_name},
        ]
    )
    message = vision_llm.invoke([hmessage])
    res = message.content
    return res


def generate_text(description, prompt):
    hmessage = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt.format(description),
            }
        ]
    )
    message = txt_llm.invoke([hmessage])
    res = message.content
    return res


def main():
    with open('questions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["filename", "instruction", "input", "output"]
        writer.writerow(field)
        for filename in os.listdir(image_path):
          #filename="IMG_4722.jpeg"
          f = os.path.join(image_path, filename)
          # checking if it is a file
          if os.path.isfile(f):
              print(f)
              q_msg = """
        Generate 50 questions someone can ask from an image with the following description. Limit the questions to the available information in the description. Just include rephrased question in response. Respond as list without bullets.
        {}
        """
              # desc = generate_description(f)
              # print(desc)
              rq_msg = """
          Rephrase the following questions for the probability of a better response in the context of an image. Use following <Example>. Provide the response as list.
          <Example>
          Question: How many burners are there?
          Rephrased Question: Do you see any burners in the image? If yes then how many burners are there?
          </Example>
          {}
          """
              ques = generate_questions(f)
              r_ques = generate_text(ques, rq_msg)
              parsed_ques = ques.split("\n")
              parsed_r_ques = r_ques.split("\n")
              if len(parsed_ques) == len(parsed_r_ques):
                for i, q in enumerate(parsed_ques):
                    input = re.sub('^\d+\.+', '', q).strip()
                    output = re.sub('^\d+\.+', '', parsed_r_ques[i]).strip()
                    writer.writerow([filename, "Rephrase the question", input, output])
              else:
                print(f, len(parsed_ques), len(parsed_r_ques), ques, r_ques)


if __name__ == "__main__":
    main()
