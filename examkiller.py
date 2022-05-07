import os
import openai
import PDFplumber 

def get_response(question):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=question,
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response.choices[0].text

def get_questions_from_pdf():
    with PDFplumber.open("document_path.PDF") as temp:
        first_page = temp.pages[0]
        print(first_page.extract_text())

if __name__ == '__main__':
    get_questions_from_pdf()