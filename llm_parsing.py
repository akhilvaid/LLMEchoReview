import os
import time
import pathlib

import fitz
import tqdm

import pandas as pd

from sklearn import metrics
from langchain.text_splitter import CharacterTextSplitter

from config import Config
from model_zoo import ModelContainer, EmbeddingModel


class EchoParser:
    @classmethod
    def read_report(cls, report_path):
        document = fitz.open(report_path)
        document_path = pathlib.Path(report_path)
        study_date = document_path.stem.split('_')[1]

        report_text = ''
        for page in document:
            report_text += page.get_text()

        return report_text, study_date

    def __init__(self):
        # Text splitter for splitting... text.
        # Values for LLAMA-30b: chunk_size=1000, chunk_overlap=0
        # Values for FastChat: chunk_size=500, chunk_overlap=100
        self.text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,  # Directly influences the size of the prompt
            chunk_overlap=100,
            length_function=len
        )

        # Load QA list
        self.df_qa = pd.read_excel(Config.qa_collection)
        self.df_qa.columns = ['subject_id', 'question', 'answer', 'optional']
        self.df_qa['subject_id'] = self.df_qa['subject_id'].astype(str)

        # Embedding model - CPU only to avoid OOM from CUDA collisions
        print('Loading embedding model...')
        self.embedding_model = EmbeddingModel.load_model()

        # For later references
        self.is_v2_llama = False

    def split_reports(self, reports):
        """Does the splitting of reports into chunks"""

        all_reports = []

        for top_count, (chartdate, note) in enumerate(reports):
            # Reduce chartdate to YYYY-MM-DD
            chartdate = chartdate.strftime('%Y-%m-%d')

            # Split the note into chunks and add order identifiers
            subnotes = self.text_splitter.split_text(note)
            for lower_count, sub_report in enumerate(subnotes):
                sub_report = f'{str(chartdate)} Report {top_count + 1} Part {lower_count + 1}: {sub_report}'
                all_reports.append(sub_report)

        return all_reports

    def load_echo_reports(self, mrn: str):
        """Returns a list of PRE-SPLIT reports for this patient"""
        report_files = [
            os.path.join(Config.dir_echo_reports, mrn, filename)
            for filename in os.listdir(os.path.join(Config.dir_echo_reports, mrn))
            if filename.endswith('.pdf')
        ]

        # Read each echo report
        report_texts = [
            self.read_report(report_path)
            for report_path in report_files
        ]

        # Sort by study date
        df_reports = pd.DataFrame(report_texts, columns=['text', 'chartdate'])
        df_reports['chartdate'] = pd.to_datetime(df_reports['chartdate'])
        df_reports = df_reports.sort_values('chartdate', ascending=True)
        these_reports = self.split_reports(df_reports[['chartdate', 'text']].values)

        return these_reports

    def similarity_search(self, question, echo_reports, embeddings):
        # Encode question
        question_embedding = self.embedding_model.encode(question)

        # Decide on total number of fragments depdending on the kind of model in use
        n_fragments = Config.num_fragments if self.is_v2_llama else Config.num_fragments // 2

        # Calculate cosine_similarity between question and each note fragment
        similarity_scores = metrics.pairwise.cosine_similarity(question_embedding.reshape(1, -1), embeddings)
        fragment_idx = similarity_scores.squeeze().argsort()[-n_fragments:][::-1]

        # Get corresponding text
        fragments = [echo_reports[fragment] for fragment in fragment_idx]

        return fragments

    def generate_prompt(self, question, fragments):
        # Prompt prefix
        prefix = 'Based on the following fragments from multiple echocardiogram reports, answer this question:'

        # Generate prompt
        fragment_text = '\n'.join(fragments)
        prompt = f'{prefix}\n{question}\n\n{fragment_text}\n\nResponse:'

        return prompt

    def question_answer(self, model_container, subject_id, qa_pairs, echo_reports, embeddings):
        # Record responses
        all_responses = {'question': [], 'answer': [], 'response': [], 'time_taken': [], 'fragments': []}

        # Start generation
        print('Starting generation...')
        for question, answer in qa_pairs:

            # Time for question
            question_start = time.time()

            # Generate prompt
            relevant_fragments = self.similarity_search(question, echo_reports, embeddings)
            prompt = self.generate_prompt(question, relevant_fragments)

            # Generate response and parse
            response = model_container.generate(prompt)
            response = response[0]['generated_text'].split(relevant_fragments[-1])[-1].strip()

            print(f'Q: {question}')
            print(f'A: {answer}')
            print(f'R: {response}')

            # Time for question
            question_end = time.time()

            # Print a red colored dashed line
            print('\033[91m' + '-' * 100 + '\033[0m')

            # Record response
            all_responses['question'].append(question)
            all_responses['answer'].append(answer)
            all_responses['response'].append(response)
            all_responses['time_taken'].append(question_end - question_start)
            all_responses['fragments'].append(relevant_fragments)

        df_responses = pd.DataFrame(all_responses)
        df_responses['subject_id'] = subject_id

        return df_responses

    def hammer_time(self, model):
        # Time to hammer
        responses_for_each_subject = []

        # Load model and tokenizer
        print(f'Loading {model.value} and tokenizer...')
        self.is_v2_llama = 'llama-2' in model.value.lower()
        if self.is_v2_llama:
            print('LLaMA-2 model detected')
        else:
            print('Non-LLaMA-2 model')

        # Init the pipeline
        model_container = ModelContainer(model, self.is_v2_llama)

        # Begin iteration
        for subject_id in tqdm.tqdm(self.df_qa['subject_id'].unique()):
            print(f'Processing subject {subject_id}...')

            # Get all QA pairs for this subject
            df_sid = self.df_qa[self.df_qa['subject_id'] == subject_id]
            qa_pairs = df_sid[['question', 'answer']].values

            # Load all echo reports for this patient
            # followed by embedding generation (ONCE)
            print('Loading echo reports and generating embeddings...')
            echo_reports = self.load_echo_reports(subject_id)
            embeddings = self.embedding_model.encode(echo_reports)

            # Generate responses using the pipeline
            df_responses = self.question_answer(model_container, subject_id, qa_pairs, echo_reports, embeddings)
            responses_for_each_subject.append(df_responses)

        # Save responses
        if responses_for_each_subject:  # If any responses were generated
            os.makedirs('Results', exist_ok=True)
            model_name_string = model.value.split('/')[-1]
            df_responses = pd.concat(responses_for_each_subject, ignore_index=True)
            df_responses.to_excel(f'Results/Responses_{model_name_string}.xlsx', index=False)
