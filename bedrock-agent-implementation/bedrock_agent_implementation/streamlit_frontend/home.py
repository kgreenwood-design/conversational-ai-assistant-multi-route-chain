import streamlit as st
import boto3
import os
import random
import string
import uuid
from datetime import datetime
from dotenv import load_dotenv
import logging
from botocore.exceptions import ClientError
import base64
import time

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Analogic Product Support AI", layout="wide")

# Load and set background image
def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
                background-size: 80% auto;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        logger.error(f"Error loading background image: {str(e)}")

if os.path.exists('image.png'):
    add_bg_from_local('image.png')
else:
    st.warning("Background image not found. Please ensure 'image.png' is in the correct directory.")

# Add custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100;300;400;500;700;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Load custom CSS
if os.path.exists('style.css'):
    try:
        with open('style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Error loading custom CSS: {str(e)}")

BEDROCK_AGENT_ALIAS = os.getenv('BEDROCK_AGENT_ALIAS')
BEDROCK_AGENT_ID = os.getenv('BEDROCK_AGENT_ID')

# Check if environment variables are set
if not BEDROCK_AGENT_ALIAS or not BEDROCK_AGENT_ID:
    st.error("BEDROCK_AGENT_ALIAS or BEDROCK_AGENT_ID environment variables are not set.")
    st.stop()

try:
    bedrock_client = boto3.client('bedrock-agent-runtime')
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('ChatHistory')
except Exception as e:
    st.error("Error initializing AWS clients. Please check your credentials and environment settings.")
    logger.error(f"Error initializing AWS clients: {str(e)}")
    st.stop()

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'show_history' not in st.session_state:
    st.session_state.show_history = False
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'feedback' not in st.session_state:
    st.session_state.feedback = {}

def format_retrieved_references(references):
    formatted_output = ""
    for reference in references:
        content_text = reference.get("content", {}).get("text", "")
        s3_uri = reference.get("location", {}).get("s3Location", {}).get("uri", "")
        formatted_output += f"Reference Information:\nContent: {content_text}\nS3 URI: {s3_uri}\n"
    return formatted_output

def process_stream(stream):
    try:
        trace = stream.get("trace", {}).get("trace", {}).get("orchestrationTrace", {})
        if trace:
            knowledgeBaseInput = trace.get("invocationInput", {}).get("knowledgeBaseLookupInput", {})
            if knowledgeBaseInput:
                logger.info(f'Looking up in knowledgebase: {knowledgeBaseInput.get("text", "")}')
            knowledgeBaseOutput = trace.get("observation", {}).get("knowledgeBaseLookupOutput", {})
            if knowledgeBaseOutput:
                retrieved_references = knowledgeBaseOutput.get("retrievedReferences", {})
                if retrieved_references:
                    return format_retrieved_references(retrieved_references)
        if "chunk" in stream:
            text = stream["chunk"]["bytes"].decode("utf-8")
            return text
    except Exception as e:
        logger.error(f"Error processing stream: {e}")

def session_generator():
    digits = ''.join(random.choice(string.digits) for _ in range(4))
    chars = ''.join(random.choice(string.ascii_lowercase) for _ in range(3))
    pattern = f"{digits[0]}{chars[0]}{digits[1:3]}{chars[1]}-{digits[3]}{chars[2]}"
    logger.info(f"Session ID: {pattern}")
    return pattern

def save_to_dynamodb(session_id, conversation, feedback=None, username=None):
    timestamp = datetime.now().isoformat()
    item = {
        'id': str(uuid.uuid4()),
        'session_id': session_id,
        'conversation': conversation,
        'timestamp': timestamp,
        'feedback': feedback
    }
    if username:
        item['username'] = username
    try:
        table.put_item(Item=item)
        logging.info("Conversation saved successfully!")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        if error_code == 'AccessDeniedException':
            logging.error(f"AccessDeniedException: {error_message}")
            logging.warning("Unable to save conversation due to permissions. Please check DynamoDB access.")
        else:
            logging.error(f"Unexpected error when saving to DynamoDB: {error_code} - {error_message}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
    return True

def ensure_dynamodb_table_exists():
    try:
        table_name = 'ChatHistory'
        existing_tables = dynamodb.meta.client.list_tables()['TableNames']
        if table_name not in existing_tables:
            table = dynamodb.create_table(
                TableName=table_name,
                KeySchema=[{'AttributeName': 'id', 'KeyType': 'HASH'}],
                AttributeDefinitions=[{'AttributeName': 'id', 'AttributeType': 'S'}],
                ProvisionedThroughput={'ReadCapacityUnits': 5, 'WriteCapacityUnits': 5}
            )
            table.meta.client.get_waiter('table_exists').wait(TableName=table_name)
            logger.info(f"Table {table_name} created successfully.")
        else:
            logger.info(f"Table {table_name} already exists.")
    except ClientError as e:
        logger.error(f"Error ensuring DynamoDB table exists: {e}")

def provide_feedback(message_index, feedback_type):
    st.session_state.feedback[message_index] = feedback_type
    username = st.session_state.get('username')
    save_to_dynamodb(st.session_state.session_id, st.session_state.conversation, st.session_state.feedback, username)
    st.success("Thank you for your feedback!")
    time.sleep(1)
    st.experimental_rerun()

def submit_question():
    if st.session_state.user_input and not st.session_state.processing:
        st.session_state.processing = True
        st.session_state.conversation.append({'user': st.session_state.user_input})
        user_input = st.session_state.user_input
        st.session_state.user_input = ""
        
        try:
            with st.spinner("Processing your request..."):
                logger.info(f"Invoking Bedrock Agent with input: {user_input}")
                response = bedrock_client.invoke_agent(
                    agentId=BEDROCK_AGENT_ID,
                    agentAliasId=BEDROCK_AGENT_ALIAS,
                    sessionId=st.session_state.session_id,
                    endSession=False,
                    inputText=user_input
                )
                logger.info("Bedrock Agent invoked successfully")
                results = response.get("completion")
                if not results:
                    logger.error("No completion in Bedrock Agent response")
                    raise ValueError("No completion in Bedrock Agent response")
                
                answer = ""
                for stream in results:
                    processed = process_stream(stream)
                    if processed:
                        answer += processed
                    else:
                        logger.warning(f"Empty processed stream: {stream}")
                
                if not answer:
                    logger.error("No answer generated from processed streams")
                    raise ValueError("No answer generated from processed streams")
                
                st.session_state.conversation.append({'assistant': answer})
                logger.info("Answer appended to conversation")
            
            save_to_dynamodb(st.session_state.session_id, st.session_state.conversation)
            logger.info("Conversation saved to DynamoDB")
        except Exception as e:
            st.error("An error occurred. Please try again later.")
            logger.error(f"Exception when calling Bedrock Agent: {str(e)}")
            logger.exception("Full traceback:")
        finally:
            st.session_state.processing = False
        
        # Force a rerun to update the UI
        st.experimental_rerun()

def main():
    st.title("Analogic Product Support AI")

    # Ensure DynamoDB table exists
    ensure_dynamodb_table_exists()

    # Initialize the agent session id if not already set
    if 'session_id' not in st.session_state:
        st.session_state.session_id = session_generator()

    # Sidebar for conversation history and controls
    st.sidebar.title("Conversation Controls")
    if st.sidebar.button("Toggle History"):
        st.session_state.show_history = not st.session_state.show_history
    if st.sidebar.button("Clear History"):
        st.session_state.conversation = []
        st.session_state.session_id = session_generator()

    # Option to reverse rendering
    reverse_rendering = st.sidebar.checkbox("Reverse Rendering")

    # Main chat interface
    chat_container = st.container()
    input_container = st.container()

    def render_chat():
        for idx, interaction in enumerate(st.session_state.conversation):
            if 'user' in interaction:
                st.markdown(f'<div class="user-message"><span style="color: #4A90E2; font-weight: bold;">You:</span> {interaction["user"]}</div>', unsafe_allow_html=True)
            elif 'assistant' in interaction:
                st.markdown(f'<div class="assistant-message"><span style="color: #50E3C2; font-weight: bold;">Assistant:</span> {interaction["assistant"]}</div>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 1, 5])
                with col1:
                    if st.button("👍", key=f"thumbs_up_{idx}"):
                        provide_feedback(idx, "positive")
                with col2:
                    if st.button("👎", key=f"thumbs_down_{idx}"):
                        provide_feedback(idx, "negative")

    def render_input():
        st.text_area("Ask a question:", key="user_input", height=50, on_change=None)
        col1, col2 = st.columns([3, 1])
        with col1:
            submit_button = st.button("Submit", key="submit_button", on_click=submit_question, use_container_width=True)
        with col2:
            with st.expander("Options", expanded=False):
                if st.button("Clear Input", key="clear_input_button", on_click=clear_input, use_container_width=True):
                    pass
                if st.button("Clear History", key="clear_history_button", use_container_width=True):
                    st.session_state.conversation = []
                    st.session_state.session_id = session_generator()
                    st.session_state.feedback = {}
                    st.experimental_rerun()

    if reverse_rendering:
        with input_container:
            render_input()
        with chat_container:
            render_chat()
    else:
        with chat_container:
            render_chat()
        with input_container:
            render_input()
    
    # Add this line to ensure the conversation is always displayed
    render_chat()

def clear_input():
    st.session_state.user_input = ""
    st.experimental_rerun()

if __name__ == '__main__':
    main()
