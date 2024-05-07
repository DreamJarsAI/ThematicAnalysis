import time
import os
from docx import Document
import textract
import pprint
from openai import OpenAI
import json
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
import requests
import os
import shutil
import zipfile

def read_transcripts(directory_path):
    """
    Function to read .docx, .doc, .pdf, or .txt files in a directory
    and extract text.

    Parameters:
    directory_path (str): Path to the directory containing the files.

    Returns:
    list: A list of strings where each string is the text extracted from a file.
    """

    transcripts = []

    for filename in tqdm(os.listdir(directory_path)):
        filepath = os.path.join(directory_path, filename)

        if filename.endswith('.docx'):
            doc = Document(filepath)
            fullText = [para.text for para in doc.paragraphs]
            transcripts.append('\n'.join(fullText))

        elif filename.endswith('.doc') or filename.endswith('.pdf') or filename.endswith('.txt'):
            try:
                text = textract.process(filepath).decode("utf-8")
            except UnicodeDecodeError:
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                        text = file.read()
                except Exception as e:
                    print(f"Error reading file {filename}: {e}")
                    continue  # Skip appending to transcripts if error occurs
            transcripts.append(text)
        else:
            print(f"Skipped unsupported file type: {filename}")

    print(f"Total number of transcripts: {len(transcripts)}")
    print(f"Total number of tokens in the transcripts:",
          sum([len(word_tokenize(transcript)) for transcript in transcripts]))

    return transcripts

def chunk_text(text, max_tokens_chunk = 2500, overlap_tokens = 100):
    """
    Function to split a long text into smaller chunks with a specified
    maximum token count.
    It also allows for an overlap between the chunks to maintain continuity
    and context between them.
    The function tokenizes the text into sentences, ensuring each chunk is
    built up with complete sentences, adhering to the specified token limits.

    Parameters:
    text (str): The input text to be chunked.
    max_tokens (int, optional): The maximum number of tokens allowed
    in each chunk. Defaults to 2500.
    overlap_tokens (int, optional): The number of tokens to be overlapped
    between adjacent chunks. Defaults to 100.

    Returns:
    list: A list of strings, where each string is a chunk of the original text
    adhering to the specified token constraints.
    """

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_token_count = 0

    for sentence in sentences:
        sentence_tokens = word_tokenize(sentence)
        sentence_token_count = len(sentence_tokens)

        if current_token_count + sentence_token_count > max_tokens_chunk:
            chunks.append(' '.join(current_chunk))

            # Creat overlap for the next chunk
            overlap_chunk = []
            overlap_token_count = 0
            while overlap_token_count < overlap_tokens and current_chunk:
                overlap_sentence = current_chunk.pop(-1)
                overlap_chunk.insert(0, overlap_sentence)
                overlap_token_count += len(word_tokenize(overlap_sentence))

            # After fulfilling the overlap requirement, set current chunk
            # to be the overlapping chunk, and append sentences behind it
            current_chunk = overlap_chunk

        current_chunk.append(sentence)
        current_token_count = len(word_tokenize(' '.join(current_chunk)))

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def process_transcripts(directory_path,
                        max_tokens_chunk = 2500, overlap_tokens = 100):
    """
    Function to process transcripts by reading files and then chunking the text.

    Parameters:
    directory_path (str): Path to the directory containing the text files.
    max_tokens_chunk (int, optional): Maximum tokens in each text chunk.
    Defaults to 2500.
    overlap_tokens (int, optional): Overlap tokens between text chunks.
    Defaults to 100.

    Returns:
    list of lists: A list where each element is a list of text chunks
    from a file.
    """

    # Read transcripts from files in the directory
    transcripts = read_transcripts(directory_path)

    # Chunk each transcript
    chunked_transcripts = [
        chunk_text(transcript, max_tokens_chunk = max_tokens_chunk,
                   overlap_tokens = overlap_tokens)
        for transcript in transcripts]

    return chunked_transcripts


def create_prompt(context, research_questions, script, text_chunk, prompt_type):
    """
    Function to create a prompt for the GPT model based on provided arguments.

    Parameters:
    context (str): Context of the study.
    research_questions (str): Research questions of the study.
    script (str): Script used in interviews or focus groups.
    text_chunk (str): A chunk of transcribed text or previously identified
    themes.
    prompt_type (str): Type of prompt - "transcript", "themes_same_id", or
    "themes_diff_id". "transcript" denotes the transcribed segment of
    an interview or focus group study from a single study participant;
    "themes_same_id" denotes the identified themes from a single study
    participant; and "themes_diff_id" denotes the identified themes from
    multiple study participants.

    Returns:
    str: A formatted string to be used as a prompt for the GPT model
    """

    if prompt_type == "transcript":
        objective = (
            "Given the extensive length of the transcripts, they have been "
            "segmented to facilitate a more manageable analysis. "
            "Your task involves three objectives, each focused on analyzing "
            "and representing the themes from the transcript segment:\n"
            "1. Identify Main Themes: Begin by discerning the main themes. "
            "For each one, formulate a concise topic sentence that "
            "encapsulates its essence.\n"
            "2. Explain the Themes: Subsequently, for each identified theme, "
            "provide a detailed explanation that elaborates on its "
            "significance and context.\n"
            "3. Illustrative Quotes: Select one impactful quote from the "
            "transcript segment that accurately exemplifies each theme. "
            "Ensure that the chosen quote is a representative embodiment of "
            "the theme it is paired with.\n"
            )
        action = ("**Transcript segment:**\n"
                  f"{text_chunk}")
        quote = ("illustrative quote")

    elif prompt_type == "themes_same_id":
        objective = (
            "Previously, you identified, elaborated, and exemplified "
            "various themes from each transcript segment. Each theme was "
            "meticulously detailed, involving a concise topic sentence, "
            "a comprehensive explanation, and a selected illustrative quote.\n"
            "Now, a more synthesized analysis is required. Having compiled "
            "the segmented analyses into a unified document, your task is "
            "to critically examine the identified themes across the segments. "
            "Assess the themes for similarities and differences, "
            "aiming to integrate and synthesize them into a more concise set "
            "of distinct themes.\n"
            "Please refine and consolidate the themes, "
            "reducing redundancy and emphasizing uniqueness and significance.\n"
            )
        action = (f"**Themes to be Synthesized:**\n"
                  f"{text_chunk}")
        quote = ("illustrative quote")

    else:
        objective = (
            "You have previously summarized various themes from each study "
            "participant's transcript. Your next task is to meticulously "
            "evaluate these themes, identifying similarities and differences "
            "across participants. Your aim should be to integrate and condense "
            "them into a more concise and distinct set of themes.\n"
            "Pay special attention to identifying and synthesizing the most "
            "prevalent themes among participants. Present the identified "
            "themes in the descending order of popularity, showcasing the most "
            "common themes first, followed by the less common ones.\n"
          	"Make sure each theme is relevant to the study's research question."
            )
        action = (f"**Themes to be Synthesized:**\n"
                  f"{text_chunk}")
        quote = ("2-3 illustrative quotes")

    if script==None:
        prompt_template = (
            "I am a university professor, seeking assistance from a research "
            "assistant. Our research team has conducted several interviews "
            "and focus group sessions, utilizing a semi-structured script, "
            "and the audio from these sessions has been transcribed into text.\n"
            "Here are the essential background details of the study:\n"
            "**Context of the Study:**\n"
            f"{context}\n"
            "**Research Questions:**\n"
            f"{research_questions}\n"
            "**Objective:**\n"
            f"{objective}"
            f"{action}\n"
            "**Format of Your Response:**\n"
            "Structure your response by delineating each theme separately "
            "and sequentially. For each theme, follow this format:\n"
            "Theme 1:\n"
            "- Topic Sentence: [Your succinct topic sentence here.]\n"
            "- Explanation: [Your comprehensive explanation here.]\n"
            f"- Quote: '[Your chosen {quote} here.]'\n"
            "Theme 2:\n"
            "- Topic Sentence: ...\n"
            "Continue in the same manner for each subsequent theme, "
            "organizing the information clearly and coherently. "
            "Please exclude unnecessary information, such as descriptions "
            "preceding each theme (e.g., Theme 1)."
            "Please aim to provide an optimal number of themes, make "
            "them as concise as possible and ensure their relevance to "
            "the study's research question."
            )
    else: 
        prompt_template = (
            "I am a university professor, seeking assistance from a research "
            "assistant. Our research team has conducted several interviews "
            "and focus group sessions, utilizing a semi-structured script, "
            "and the audio from these sessions has been transcribed into text.\n"
            "Here are the essential background details of the study:\n"
            "**Context of the Study:**\n"
            f"{context}\n"
            "**Research Questions:**\n"
            f"{research_questions}\n"
            "**Script:**\n"
            f"{script}\n"
            "**Objective:**\n"
            f"{objective}"
            f"{action}\n"
            "**Format of Your Response:**\n"
            "Structure your response by delineating each theme separately "
            "and sequentially. For each theme, follow this format:\n"
            "Theme 1:\n"
            "- Topic Sentence: [Your succinct topic sentence here.]\n"
            "- Explanation: [Your comprehensive explanation here.]\n"
            f"- Quote: '[Your chosen {quote} here.]'\n"
            "Theme 2:\n"
            "- Topic Sentence: ...\n"
            "Continue in the same manner for each subsequent theme, "
            "organizing the information clearly and coherently. "
            "Please exclude unnecessary information, such as descriptions "
            "preceding each theme (e.g., Theme 1)."
            "Please aim to provide an optimal number of themes, make "
            "them as concise as possible and ensure their relevance to "
            "the study's research question."
            )

    return prompt_template


def generate_response(prompt, api_key, model = "gpt-4", max_tokens = 1000):
    """
    Function to get a response from the OpenAI GPT-4 model.

    Parameters:
    prompt (str): The prompt that will be sent to the model for completion.
    api_key (str): The API key for the OpenAI API.
    model (str, optional): The model to be used for completion.
    Default to "gpt-4".
    max_tokens (int, optional): The maximum length of the generated text.
    Defaults to 1000.

    Returns:
    str: The generated text from the model.
    """
    client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=api_key,
    )
    response = client.chat.completions.create(
        model = model,
        messages = [
            {
                "role": "system",
                "content": "You are a helpful research assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens = max_tokens
    )

    # OLD configuration

    # # Configure the OpenAI API client
    # openai.api_key = api_key

    # # Make a request to the OpenAI API using the chat endpoint
    # response = openai.ChatCompletion.create(
    #     model = model,
    #     messages = [
    #         {
    #             "role": "system",
    #             "content": "You are a helpful research assistant."
    #         },
    #         {
    #             "role": "user",
    #             "content": prompt
    #         }
    #     ],
    #     max_tokens = max_tokens
    # )
    # return response['choices'][0]['message']['content']
    # Return the model's response
    return response.choices[0].message.content
    


def analyze_transcripts(
    directory_path, context, research_questions, script,
    api_key, model = "gpt-4", max_tokens_transcript = 1000,
    max_tokens_chunk = 2500, overlap_tokens = 100):
    """
    Function to analyze transcripts using a GPT model.

    Parameters:
    directory_path (str): Path to the directory containing the text files.
    context (str): Context of the study.
    research_questions (str): Research questions of the study.
    script (str): Script used in interviews or focus groups.
    api_key (str): The API key for the OpenAI API.
    model (str, optional): The model to be used for completion.
    Defaults to "gpt-4".
    max_tokens_transcript (int, optional): The maximum length of the generated
    text from a transcript segment. Defaults to 1000.
    max_tokens_chunk (int, optional): The maximum length of a transcript
    segment. Defaults to 2500.
    overlap_tokens (int, optional): The number of tokens to overlap between
    transcript segments. Defaults to 100.

    Returns:
    list of lists: A list where each element is a model-generated text
    for a chunk.
    """

    # Caculate the number of tokens in context, research questions, and script
    tokens_in_prompt = len(word_tokenize(context)) + \
    len(word_tokenize(research_questions)) + len(word_tokenize(script))

    # Check if the prompt is too long
    if tokens_in_prompt > 1600:
      raise ValueError("""The prompt is too long. \n
      Please reconstruct the context, research questions, and script so that
      their total number of tokens to be below 1600 tokens.""")

    # Process the transcripts to get chunks
    transcript_chunks = process_transcripts(
        directory_path = directory_path, max_tokens_chunk = max_tokens_chunk,
        overlap_tokens = overlap_tokens)

    # Count the total number of chunks
    total_chunks = sum(len(chunks) for chunks in transcript_chunks)
    print(f"Total number of transcript chunks to be processed: {total_chunks}")

    # Create a tqdm object
    pbar = tqdm(total = total_chunks, unit = "chunk", desc = "Processing",
                dynamic_ncols = True)

    # Analyze each chunk
    analysis_results = []
    for chunks in transcript_chunks:
        chunk_results = []
        for chunk in chunks:
            # Create a prompt for each chunk
            prompt = create_prompt(
                context = context, research_questions = research_questions,
                script = script, text_chunk = chunk, prompt_type = "transcript")

            # Get model-generated response
            response = generate_response(
                prompt = prompt, api_key = api_key, model = model,
                max_tokens = max_tokens_transcript)
            chunk_results.append(response)

            # Update tqdm object
            pbar.update(1)

        analysis_results.append(chunk_results)

    # Close tqdm object when done
    pbar.close()

    print("Total number of tokens generated from analyzing transcript chunks:",
          sum([sum([len(word_tokenize(result)) for result in chunk_results])
          for chunk_results in analysis_results]))

    return analysis_results


def combine_themes(themes_list, token_threshold = 5000):
    """
    Function to combine the text chunks respecting a token limit.

    Parameters:
    themes_list (list of str): Each element is a string of text.
    token_threshold (int, optional): The maximum number of tokens for each
    combined text. Defaults to 5000.

    Returns:
    list of str: A list containing combined texts each with tokens up
    to the token threshold.
    """

    combined_texts = []
    current_text = ""
    current_tokens = 0

    for text in themes_list:
        tokens = word_tokenize(text)
        text_tokens = len(tokens)

        if current_tokens + text_tokens > token_threshold:
            combined_texts.append(current_text.strip())
            current_text = text
            current_tokens = text_tokens
        else:
            current_text += " " + text
            current_tokens += text_tokens

    if current_text:  # Adding any remaining text
        combined_texts.append(current_text.strip())

    return combined_texts


def synthesize_themes(themes_list, context, research_questions, script,
                      api_key, prompt_type = "themes_same_id",
                      model = "gpt-4", max_tokens_combine = 5000,
                      max_tokens_gen_themes = 1000):
    """
    Function to recursively synthesize and refine themes using a GPT model.
    The function takes a list of themes (each a string containing a chunk of
    text), combines and analyzes them, repeating this process until a single
    consolidated chunk of themes remains.

    Parameters:
    themes_list (list): A list containing strings of text chunks to be
    synthesized.
    context (str): Context of the study.
    research_questions (str): Research questions of the study.
    script (str): Script used in interviews or focus groups.
    api_key (str): The API key for accessing the GPT model.
    prompt_type (str, optional): The type of prompt to be used.
    Defaults to "themes_same_id". If set to "themes_diff_id", the function will
    recursively analyze and refine themes from different study participants.
    model (str, optional): The specific GPT model to be used.
    Defaults to "gpt-4".
    max_tokens_combine (int, optional): Maximum token length for
    combined themes. Defaults to 5000.
    max_tokens_gen_themes (int, optional): Maximum tokens for generated
    responses. Defaults to 1000.

    Returns:
    list: A list containing a single string, which is the synthesized
    chunk of themes.
    """

    # Combine the themes into manageable chunks
    combined_themes = combine_themes(themes_list = themes_list,
                                     token_threshold = max_tokens_combine)

    new_themes = []
    for theme in combined_themes:
        # Create a prompt for each theme.
        prompt = create_prompt(
            context = context, research_questions = research_questions,
            script = script, text_chunk = theme, prompt_type = prompt_type)

        # Get model-generated response.
        response = generate_response(prompt = prompt, api_key = api_key,
                                     model = model,
                                     max_tokens = max_tokens_gen_themes)
        new_themes.append(response)

    # Check if further processing is needed.
    if len(new_themes) == 1:
        return new_themes
    else:
        return synthesize_themes(
            themes_list = new_themes, context = context,
            research_questions = research_questions, script = script,
            api_key = api_key, prompt_type = prompt_type, model = model,
            max_tokens_combine = max_tokens_combine,
            max_tokens_gen_themes = max_tokens_gen_themes)
    

def save_results_to_json(results, file_path):
    """
    Save the analysis results to a JSON file.

    Parameters:
    results (list): The analysis results to save.
    file_path (str): The path where the JSON file will be saved.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, 'w', encoding = 'utf-8') as file:
        json.dump(results, file, ensure_ascii = False, indent = 0)


def load_results_from_json(file_path):
    """
    Load the analysis results from a JSON file.

    Parameters:
    file_path (str): The path where the JSON file is located.

    Returns:
    list: The loaded analysis results.
    """
    with open(file_path, 'r', encoding = 'utf-8') as file:
        results = json.load(file)
    return results

def analyze_and_synthesize_transcripts(
    directory_path, context, research_questions, script, api_key,
    save_results_path, model = "gpt-4",
    max_tokens_chunk = 2500, overlap_tokens = 100,
    max_tokens_transcript = 1000, max_tokens_combine_ind_themes = 5000,
    max_tokens_gen_ind_themes = 1000, max_tokens_combine_all_themes = 4000,
    max_tokens_gen_all_themes = 2000):
    """
    Function to analyze transcripts, synthesize individual themes,
    and synthesize overall study themes using a GPT model.

    Parameters:
    directory_path (str): Path to the directory containing the text files.
    context (str): Context of the study.
    research_questions (str): Research questions of the study.
    script (str): Script used in interviews or focus groups.
    api_key (str): The API key for the OpenAI API.
    save_results_path (str): The path where the results will be saved.
    The two results are "themes_per_person.json" and "themes_overall.json",
    which contain the results of Step 2 and Step 3, respectively.
    model (str, optional): The model to be used for completion.
    Defaults to "gpt-4".
    max_tokens_chunk (int, optional): The maximum number of tokens in each
    transcript segment. Defaults to 2500.
    overlap_tokens (int, optional): The number of tokens that overlap between
    two consecutive transcript segments. Defaults to 100.
    max_tokens_transcript (int, optional): The maximum length of the generated
    text from a transcript segment. Defaults to 1000.
    max_tokens_combine_ind_themes (int, optional): Maximum token length for
    combining an single participant's themes. Defaults to 5000.
    max_tokens_gen_ind_themes (int, optional): Maximum tokens for generating
    themes from a chunk of combined themes from a single participant.
    Defaults to 1000.
    max_tokens_combine_all_themes (int, optional): Maximum token length for
    combining themes from all participants. Defaults to 4000.
    max_tokens_gen_all_themes (int, optional): Maximum tokens for generating
    themes from all participants (each individual has a single chunck of
    themes). Defaults to 2000.

    Returns:
    list of lists: A list where each sublist contains a synthesized chunk of
    overall study themes.
    """

    # Record the start time
    start_time = time.time()

    # Step 1: Analyze transcripts to extract initial themes
    initial_themes = analyze_transcripts(
        directory_path = directory_path, context = context,
        research_questions = research_questions, script = script,
        api_key = api_key, model = model,
        max_tokens_transcript = max_tokens_transcript,
        max_tokens_chunk = max_tokens_chunk, overlap_tokens = overlap_tokens)

    # Save the results of Step 1 to a JSON file
    save_results_to_json(
        initial_themes, os.path.join(save_results_path, "themes_raw.json"))

    # Step 2: Synthesize themes at the individual level
    individual_synthesized_themes = []
    pbar = tqdm(total = len(initial_themes),
                desc = "Synthesizing Individual Themes",
                dynamic_ncols = True)
    for themes in initial_themes:
        synthesized = synthesize_themes(
            themes_list = themes, context = context,
            research_questions = research_questions, script = script,
            api_key = api_key, model = model, prompt_type = "themes_same_id",
            max_tokens_combine = max_tokens_combine_ind_themes,
            max_tokens_gen_themes = max_tokens_gen_ind_themes)
        individual_synthesized_themes.append(synthesized)
        pbar.update(1)
    pbar.close()

    print("Total number of tokens generated from synthesizing themes for each "
          "individual:",
          sum([sum([len(word_tokenize(result)) for result in ind_results])
          for ind_results in individual_synthesized_themes]))

    # Save the results of Step 2 to a JSON file
    save_results_to_json(
        individual_synthesized_themes,
        os.path.join(save_results_path, "themes_per_person.json"))

    # Step 3: Synthesize themes at the overall study level
    all_themes = [
        theme for sublist in individual_synthesized_themes for theme in sublist]
    pbar = tqdm(total = 1, desc = "Synthesizing Overall Themes",
                dynamic_ncols = True)
    overall_synthesized_themes = synthesize_themes(
        themes_list = all_themes, context = context,
        research_questions = research_questions, script = script,
        api_key = api_key, model = 'gpt-4', prompt_type = "themes_diff_id",
        max_tokens_combine = max_tokens_combine_all_themes,
        max_tokens_gen_themes = max_tokens_gen_all_themes)
    pbar.update(1)
    pbar.close()

    print("Total number of tokens generated from synthesizing themes from all "
          "individuals:",
          sum([len(word_tokenize(result)) for result in all_themes]))

    # Save the results of Step 3 to a JSON file
    save_results_to_json(
        overall_synthesized_themes,
        os.path.join(save_results_path, "themes_overall.json"))

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Function execution time: {elapsed_time:.2f} seconds")

    return (initial_themes, individual_synthesized_themes,
            overall_synthesized_themes)

def move_contents_to_parent(source, destination):
    for item in os.listdir(source):
        s = os.path.join(source, item)
        d = os.path.join(destination, item)
        shutil.move(s, d)

def download_example(folder_name="example_transcripts",url="https://sites.wustl.edu/alba/files/2024/04/book_empathy_club-02119c68e92058fe.zip"):
    """
    Function to download example transcripts from a url

    Parameters:
    folder_name (str): Name of folder where the example transcripts will be contained in exisiting directory
    url (str): download url of the transcripts, should end in .zip format. 
    """
    citation='''
    Henderson, R., Hagen, M. G., Zaidi, Z., Dunder, V., Maska, E., & Nagoshi, Y. (2020). 
    Self-care perspective taking and empathy in a student-faculty book club in the United States. 
    Journal of Educational Evaluation for Health Professions, 17.
    '''
    print("Citation: ",citation)
    # Get the current working directory
    current_directory = os.getcwd()
    # Path for the new folder
    folder_path = os.path.join(current_directory, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Download the file
    local_zip_path = os.path.join(current_directory, "downloaded_file.zip")
    print(f"Downloading file to {local_zip_path}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(local_zip_path, 'wb') as f:
            f.write(response.content)

        # Unzip the file directly into the specified folder
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(folder_path)
        os.remove(local_zip_path)
        source_folder = os.path.join(folder_path, "book_empathy_club")
        move_contents_to_parent(source_folder, folder_path)
        if os.path.exists(source_folder) and not os.listdir(source_folder):
            os.rmdir(source_folder)
        else:
            print("Note: Source folder is not empty or does not exist.")
        print(f"Files downloaded to {local_zip_path}...")
    else:
        print("Failed to download the file. Status code:", response.status_code)

    
