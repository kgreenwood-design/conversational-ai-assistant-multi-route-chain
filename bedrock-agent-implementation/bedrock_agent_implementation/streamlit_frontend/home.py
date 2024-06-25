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
from boto3.dynamodb.conditions import Key

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Analogic Product Support- Development", layout="wide")

# Load and display logo
def add_logo(image_file):
    try:
        st.markdown(
            f"""
            <div class="logo-container">
                <img src="data:image/png;base64,{base64.b64encode(open(image_file, "rb").read()).decode()}" class="logo">
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        logger.error(f"Error loading logo image: {str(e)}")

if os.path.exists('image.png'):
    add_logo('image.png')
else:
    st.warning("Logo image not found. Please ensure 'image.png' is in the correct directory.")

# Center the title
st.markdown("<h1 style='text-align: center;'>Analogic Product Support- Development</h1>", unsafe_allow_html=True)

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
    table_name = os.environ.get('DYNAMODB_TABLE_NAME')
    if not table_name:
        st.error("DYNAMODB_TABLE_NAME environment variable is not set")
        st.stop()
    table = dynamodb.Table(table_name)
except Exception as e:
    st.error("Error initializing AWS clients. Please check your credentials and environment settings.")
    logger.error(f"Error initializing AWS clients: {str(e)}")
    st.stop()

# Function to clear session state
def clear_session_state():
    st.session_state.conversation = []
    st.session_state.session_id = None
    st.session_state.show_history = False
    st.session_state.user_input = ""
    st.session_state.processing = False
    st.session_state.feedback = {}

# Initialize session state
if 'initialized' not in st.session_state or st.experimental_get_query_params().get('refresh'):
    clear_session_state()
    st.session_state.initialized = True
    # Remove the 'refresh' query parameter
    params = st.experimental_get_query_params()
    params.pop('refresh', None)
    st.experimental_set_query_params(**params)

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
        'session_id': session_id,
        'timestamp': timestamp,
        'conversation': conversation
    }
    if feedback is not None:
        item['feedback'] = {str(k): str(v) for k, v in feedback.items()}
    if username:
        item['username'] = username
    try:
        table.put_item(Item=item)
        logger.info("Conversation saved successfully!")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        if error_code == 'AccessDeniedException':
            logger.error(f"AccessDeniedException: {error_message}")
            st.error("Unable to save conversation due to permissions. Please check DynamoDB access.")
        elif error_code == 'ResourceNotFoundException':
            logger.error(f"ResourceNotFoundException: {error_message}")
            st.error("DynamoDB table not found. Please check if the table exists.")
        else:
            logger.error(f"Unexpected error when saving to DynamoDB: {error_code} - {error_message}")
            st.error(f"An error occurred while saving the conversation: {error_code}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        st.error(f"An unexpected error occurred: {str(e)}")
    return True

def ensure_dynamodb_table_exists():
    try:
        table_name = os.environ.get('DYNAMODB_TABLE_NAME')
        if not table_name:
            logger.error("DYNAMODB_TABLE_NAME environment variable is not set")
            return
        
        existing_tables = dynamodb.meta.client.list_tables()['TableNames']
        if table_name not in existing_tables:
            logger.error(f"Table {table_name} does not exist. It should be created by the CDK stack.")
        else:
            logger.info(f"Table {table_name} exists.")
    except ClientError as e:
        logger.error(f"Error checking DynamoDB table: {e}")

def provide_feedback(message_index, feedback_type):
    st.session_state.feedback[message_index] = feedback_type
    username = st.session_state.get('username')
    save_to_dynamodb(st.session_state.session_id, st.session_state.conversation, st.session_state.feedback, username)
    st.success("Thank you for your feedback!")

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
        
        # Instead of forcing a rerun, we'll use st.empty() to update the UI
        st.empty()

def render_chat():
    for idx, interaction in enumerate(st.session_state.conversation):
        if 'user' in interaction:
            st.markdown(f'<div class="user-message"><span style="color: #4A90E2; font-weight: bold;">You:</span> {interaction["user"]}</div>', unsafe_allow_html=True)
        elif 'assistant' in interaction:
            st.markdown(f'<div class="assistant-message"><span style="color: #50E3C2; font-weight: bold;">Assistant:</span> {interaction["assistant"]}</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 1, 5])
            with col1:
                if st.button("👍", key=f"thumbs_up_{idx}_{len(st.session_state.conversation)}"):
                    provide_feedback(idx, "positive")
            with col2:
                if st.button("👎", key=f"thumbs_down_{idx}_{len(st.session_state.conversation)}"):
                    provide_feedback(idx, "negative")

def render_sidebar_history():
    st.sidebar.title("Conversation History")
    for idx, interaction in enumerate(st.session_state.conversation):
        if 'user' in interaction:
            st.sidebar.text(f"You: {interaction['user'][:30]}...")
        elif 'assistant' in interaction:
            st.sidebar.text(f"Assistant: {interaction['assistant'][:30]}...")

def main():
    # Title is now set in st.set_page_config, so we can remove this line

    # Ensure DynamoDB table exists
    ensure_dynamodb_table_exists()

    # Add refresh button
    if st.sidebar.button("Refresh Page"):
        st.experimental_set_query_params(refresh='true')
        st.experimental_rerun()

    # Clear session state when the app starts
    if st.sidebar.button("New Conversation"):
        clear_session_state()

    # Initialize the agent session id if not already set
    if st.session_state.session_id is None:
        st.session_state.session_id = session_generator()

    # Sidebar for conversation history and controls
    st.sidebar.title("Conversation Controls")
    if st.sidebar.button("Clear History"):
        st.session_state.conversation = []
        st.session_state.session_id = session_generator()
        st.session_state.feedback = {}
        st.experimental_rerun()

    # Render sidebar history
    render_sidebar_history()

    # Main chat interface
    chat_container = st.container()
    input_container = st.container()

    def render_input():
        st.text_area("Ask a question:", key="user_input", height=50, on_change=None)
        col1, col2 = st.columns([3, 1])
        with col1:
            submit_button = st.button("Submit", key="submit_button", on_click=submit_question, use_container_width=True)
        with col2:
            with st.expander("Options", expanded=False):
                if st.button("Clear Input", key="clear_input_button", on_click=clear_input, use_container_width=True):
                    pass

    with chat_container:
        render_chat()
    with input_container:
        render_input()

    # Add a button to verify DynamoDB entries
    if st.sidebar.button("Verify DynamoDB Entries"):
        verify_dynamodb_entries()

def clear_input():
    st.session_state.user_input = ""

def verify_dynamodb_entries():
    try:
        # Query the latest entries from DynamoDB
        response = table.query(
            KeyConditionExpression=Key('session_id').eq(st.session_state.session_id),
            ScanIndexForward=False,
            Limit=5
        )
        
        if response['Items']:
            st.sidebar.subheader("Latest DynamoDB Entries")
            for item in response['Items']:
                st.sidebar.json(item)
        else:
            st.sidebar.warning("No entries found for the current session.")
    except Exception as e:
        st.sidebar.error(f"Error querying DynamoDB: {str(e)}")
        logger.error(f"Error querying DynamoDB: {str(e)}")

if __name__ == '__main__':
    main()
