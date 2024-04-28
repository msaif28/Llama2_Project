import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, pipeline
import os
import textract
from tika import tika, parser
tika.TikaJarPath = os.path.expanduser("~")
import configs

# Function to get all supported files in the projects folder and its subfolders
def get_files(path):
    """
    Retrieves all supported files in the specified folder and its subfolders

    Args:
        path (str): Path to the folder containing the files

    Returns:
        list: List of paths to supported files
    """
     
    try:
        supported_formats = [".pdf", ".docx", ".ipynb", ".py", ".md", ".pptx", ".xls", ".xlsx", ".csv"]
        supported_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(tuple(supported_formats)):
                    supported_files.append(os.path.join(root, file))
        return supported_files
    except Exception as e:
        print(e)
        return []

# Function to parse multiple files
def parse_files(files):
    """
    Parses multiple files and returns concatenated text

    Args:
        files (list): List of file paths to be parsed

    Returns:
        tuple: Concatenated text and number of files successfully parsed
    """
    texts = ""
    i = 0
    for file in files:
        try:
            parsed_text = parse_file(file)
            print("%s parsed successfully" % file)
            texts += "\n" + parsed_text
            i += 1
        except Exception as e:
            print("%s failed to parse" % file)
            print(e)
    return texts, i

# Function to create the vector database with Sentence Transformers
def create_vector_db(texts):
    """
    Creates a vector database with Sentence Transformers for text embeddings.

    Args:
        texts (str): Text data to be processed.

    Returns:
        None
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    split_texts = text_splitter.split_text(texts)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cuda'})
    db = FAISS.from_texts(split_texts, embeddings)
    db.save_local(configs.DB_FAISS_PATH)

# Function to load the model and tokenizer
def load_model_and_tokenizer(model_name_or_path):
    """
    Loads the language model and tokenizer for the pipeline.

    Args:
        model_name_or_path (str): Name or path of the pre-trained model.

    Returns:
        pipeline: Initialized text-generation pipeline.
    """

    # Load model and tokenizer 
    model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True,
                                            trust_remote_code=False, safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
    pipe = pipeline("text-generation", model=model.model, tokenizer=tokenizer, max_new_tokens=512,
                do_sample=True, temperature=0.7, top_p=0.95, top_k=40, repetition_penalty=1.1)
    return pipe

# Function to retrieve context from the database
def get_context_from_db(query, db):
    """
    Retrieves context from the database for a given query.

    Args:
        query (str): User query.
        db: Vector database instance.

    Returns:
        context: Retrieved context.
    """
    retriever = db.as_retriever(search_kwargs={'k': 5},return_source_documents=True)
    context = retriever.get_relevant_documents(query)
    return context

# Function to generate response using your LLM approach
def generate_response_with_llm(context, questions, pipeline):
    """
    Generates a response using the Language Model (LLM) approach.

    Args:
        context: Contextual information for generating the response.
        questions (str): User's question.
        pipeline: Initialized text-generation pipeline.

    Returns:
        str: Generated response text.
    """
    # prompt_template = f'''[INST] <<SYS>> ... {context}.\n {questions} <</SYS>>[/INST]'''
    prompt_template=f'''[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. You have to answer the questions based on the context that you read. If a question does not make any sense, or is not mentioned in the context, 
    explain why instead of answering something not correct. If you don't know the answer to a question or it's not related to the project, please don't share false information and say honestly. 
    Here is the context for this conversation: \n
    {context}.\n According to this context, answer this question:
    <</SYS>>
    {questions}
    [/INST]
    '''
    
    generated_text = pipeline(prompt_template)[0]['generated_text']
    return generated_text.split('[/INST]')[-1].strip()

# QA
def question_answering(user_query, pipeline, db):
    """
    Performs question answering based on the user query and context.

    Args:
        user_query (str): User's question/query.
        pipeline: Initialized text-generation pipeline.
        db: Vector database instance.

    Returns:
        str: Generated response to the user query.
    """
    print("getting context from db...")
    context = get_context_from_db(user_query, db)

    response = generate_response_with_llm(context, user_query, pipeline)
    print("Generating Responce")
    return response


def parse_file(file_name):
    """
    Parses a file based on its extension using different libraries.

    Args:
        file_name (str): Name of the file to be parsed.

    Raises:
        Exception: If file type is not supported or parsing fails.

    Returns:
        str: Parsed text data from the file.
    """

    # Parse the file based on the file extension.
    # pdf, .docx, .ipynb, .py, .md, .pptx, .xls, .xlsx, .csv.
    
    textract_file_types = [".pptx", ".docx", ".xlsx", ".xls", ".csv"]
    tika_file_types = [".pdf"]
    python_standard_file_types = [".ipynb", ".py", ".md"]

    if file_name.endswith(tuple(textract_file_types)):
        try:
            # Parse using textract
            text = textract.process(file_name).decode("utf-8")
            return text
        except Exception as e:
            raise Exception("Failed to parse file using textract") from e
    if file_name.endswith(tuple(tika_file_types)):
        try:
            # Parse using tika
            parsed_pdf = parser.from_file(file_name)
            data = parsed_pdf['content']
            return data
        except Exception as e:
            raise Exception("Failed to parse file using tika") from e
    if file_name.endswith(tuple(python_standard_file_types)):
        try:
            # Parse using python standard library
            with open(file_name, "r") as f:
                data = f.read()
            return data
        except Exception as e:
            raise Exception("Failed to parse file using python standard library") from e
    else:
        raise Exception("File type not supported")


def streamlit_parse_file(uploaded_file):
    """
    Parses a file uploaded in a Streamlit application.

    Args:
        uploaded_file: File uploaded through the Streamlit interface.

    Raises:
        Exception: If file type is not supported or parsing fails.

    Returns:
        str: Parsed text data from the uploaded file.
    """
    # Parse the file based on the file extension.
    # pdf, .docx, .ipynb, .py, .md, .pptx, .xls, .xlsx, .csv.
    
    textract_file_types = ["pptx", "docx", "xlsx", "xls", "csv"]
    tika_file_types = ["pdf"]
    python_standard_file_types = ["ipynb", "py", "md"]

    try:
        file_extension = uploaded_file.name.split(".")[-1].lower()

        if file_extension in textract_file_types:
            # Parse using textract
            text = textract.process(uploaded_file).decode("utf-8")
            return text
        elif file_extension in tika_file_types:
            # Parse using tika
            parsed_pdf = parser.from_buffer(uploaded_file.read())
            data = parsed_pdf['content']
            return data
        elif file_extension in python_standard_file_types:
            # Parse using python standard library
            uploaded_file.seek(0)  # Reset file pointer to the beginning
            data = uploaded_file.read().decode("utf-8")
            return data
        else:
            raise Exception("File type not supported")
    except Exception as e:
        raise Exception(f"Failed to parse file: {str(e)}")
