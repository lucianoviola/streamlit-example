import streamlit as st
import requests
from io import StringIO
from transformer import pipeline

model_id = "philschmid/bart-large-cnn-samsum"
api_token = 'api_zvZqQAtBgkyqKLdHyhulHRlmwSECJVSSjo'

summarizer = pipeline("summarization", model="lidiya/bart-large-xsum-samsum")


"""
# Meeting summarizer
"""


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


@st.cache
def query(payload, model_id, api_token):
    headers = {"Authorization": f"Bearer {api_token}"}
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def main():
    path = st.file_uploader("Upload transcription", type=['csv', 'txt'])
    if not path:
        st.write("Upload a .csv or .xlsx file to get started")
        return

    stringio = StringIO(path.getvalue().decode("utf-8"))
    string_data = stringio.read()
    #st.write('Running summarizer...')
    string_data = [x for x in string_data.splitlines() if not x.startswith('Customer :')]
    #string_data = string_data.replace('Customer\n', ' ')
    #results = query(string_data, model_id, api_token)
    chunksize = round(len(string_data) / 2)
    results = [query(a,  model_id, api_token) for
               a in list(chunks(string_data, chunksize))]
    if results:
        st.write('Summary:')
        st.write(results[0]['summary_text'])

main()


