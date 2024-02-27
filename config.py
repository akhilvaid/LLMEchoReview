# General config

import textwrap


class Config:
    # Paths
    dir_model_path = ''  # LLaMA 2 cache path
    dir_echo_reports = 'EchoReports'
    dir_model_cache = '/local/tmp'  # For MINERVA only

    qa_collection = 'Data/EchoQuestions.xlsx'
    final_eval = 'EVAL_Llama-2-70b-chat-hf_eval.xlsx'  # Final evaluation file

    # Generation specific
    num_fragments = 6  # Pull these many fragments from echo reports

    # Proompting
    system_prompt = textwrap.dedent("""
    You are a professor of Cardiology. Your answers to the following question must be as accurate, informative, and to the point as possible.
    Do not include any information that is not directly relevant to the question, or offer any information that is not asked for.
    If you don't know the answer, just say so. Do not make anything up.""")
