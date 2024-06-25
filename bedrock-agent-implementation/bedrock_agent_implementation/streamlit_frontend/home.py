import streamlit as st
import boto3
import os
import random
import string
import yaml
import streamlit_authenticator as stauth
import uuid
from datetime import datetime
from dotenv import load_dotenv
import logging
from botocore.exceptions import ClientError

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Plant Technician AI", layout="wide")

BEDROCK_AGENT_ALIAS = os.getenv('BEDROCK_AGENT_ALIAS')
BEDROCK_AGENT_ID = os.getenv('BEDROCK_AGENT_ID')

# Check if environment variables are set
if not BEDROCK_AGENT_ALIAS or not BEDROCK_AGENT_ID:
    st.error("BEDROCK_AGENT_ALIAS or BEDROCK_AGENT_ID environment variables are not set.")
    st.stop()

bedrock_client = boto3.client('bedrock-agent-runtime')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('ChatHistory')

# Load configuration file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

# Create an authentication object
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

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

def format_retrieved_references(references):
    # Extracting the text and link from the references
    for reference in references:
        content_text = reference.get("content", {}).get("text", "")
        s3_uri = reference.get("location", {}).get("s3Location", {}).get("uri", "")

        # Formatting the output
        formatted_output = "Reference Information:\n"
        formatted_output += f"Content: {content_text}\n"
        formatted_output += f"S3 URI: {s3_uri}\n"

        return formatted_output


def process_stream(stream):
    try:
        # print("Processing stream...")
        trace = stream.get("trace", {}).get("trace", {}).get("orchestrationTrace", {})

        if trace:
            # print("This is a trace")
            knowledgeBaseInput = trace.get("invocationInput", {}).get(
                "knowledgeBaseLookupInput", {}
            )
            if knowledgeBaseInput:
                print(
                    f'Looking up in knowledgebase: {knowledgeBaseInput.get("text", "")}'
                )
            knowledgeBaseOutput = trace.get("observation", {}).get(
                "knowledgeBaseLookupOutput", {}
            )
            if knowledgeBaseOutput:
                retrieved_references = knowledgeBaseOutput.get(
                    "retrievedReferences", {}
                )
                if retrieved_references:
                    print("Formatted References:")
                    return format_retrieved_references(retrieved_references)

        # Handle 'chunk' data
        if "chunk" in stream:
            print("This is the final answer:")
            text = stream["chunk"]["bytes"].decode("utf-8")
            return text

    except Exception as e:
        print(f"Error processing stream: {e}")
        print(stream)

def session_generator():
    # Generate random characters and digits
    digits = ''.join(random.choice(string.digits) for _ in range(4))  # Generating 4 random digits
    chars = ''.join(random.choice(string.ascii_lowercase) for _ in range(3))  # Generating 3 random characters
    
    # Construct the pattern (1a23b-4c)
    pattern = f"{digits[0]}{chars[0]}{digits[1:3]}{chars[1]}-{digits[3]}{chars[2]}"
    print("Session ID: " + str(pattern))

    return pattern

def save_to_dynamodb(username, session_id, conversation):
    timestamp = datetime.now().isoformat()
    item = {
        'id': str(uuid.uuid4()),
        'username': username,
        'session_id': session_id,
        'conversation': conversation,
        'timestamp': timestamp
    }
    try:
        table.put_item(Item=item)
        st.success("Conversation saved successfully!")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        if error_code == 'AccessDeniedException':
            logging.error(f"AccessDeniedException: {error_message}")
            logging.warning("Unable to save conversation due to permissions. Please check DynamoDB access.")
        else:
            logging.error(f"Unexpected error when saving to DynamoDB: {error_code} - {error_message}")
        # Don't show the error to the user, just log it
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
    # Always return True to indicate the conversation was processed
    return True

def ensure_dynamodb_table_exists():
    try:
        dynamodb = boto3.resource('dynamodb')
        table_name = 'ChatHistory'
        
        # Check if the table exists
        existing_tables = dynamodb.meta.client.list_tables()['TableNames']
        if table_name not in existing_tables:
            # Create the table
            table = dynamodb.create_table(
                TableName=table_name,
                KeySchema=[
                    {'AttributeName': 'id', 'KeyType': 'HASH'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'id', 'AttributeType': 'S'}
                ],
                ProvisionedThroughput={
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            )
            # Wait for the table to be created
            table.meta.client.get_waiter('table_exists').wait(TableName=table_name)
            print(f"Table {table_name} created successfully.")
        else:
            print(f"Table {table_name} already exists.")
    except ClientError as e:
        print(f"Error ensuring DynamoDB table exists: {e}")

def main():
    st.title("Plant Technician AI")

    def submit_question():
        if st.session_state.user_input and not st.session_state.processing:
            st.session_state.processing = True
            try:
                # Add the user's prompt to the conversation state
                st.session_state.conversation.append({'user': st.session_state.user_input})

                # Format and add the answer to the conversation state
                with st.spinner("Processing your request..."):
                    response = bedrock_client.invoke_agent(
                        agentId=BEDROCK_AGENT_ID,
                        agentAliasId=BEDROCK_AGENT_ALIAS,
                        sessionId=st.session_state.session_id,
                        endSession=False,
                        inputText=st.session_state.user_input
                    )
                    results = response.get("completion")
                    answer = ""
                    for stream in results:
                        answer += process_stream(stream)
                    st.session_state.conversation.append({'assistant': answer})

                # Save the conversation to DynamoDB
                save_to_dynamodb(st.session_state.username, st.session_state.session_id, st.session_state.conversation)

                # Clear the input box after submission
                st.session_state.user_input = ""
                
            except Exception as e:
                st.error("An error occurred. Please try again later.")
                logging.error(f"Exception when calling Bedrock Agent: {e}")
            finally:
                st.session_state.processing = False
                st.experimental_rerun()

    # Ensure DynamoDB table exists
    ensure_dynamodb_table_exists()

    # Authentication
    name, authentication_status, username = authenticator.login(fields={'form_name': 'Login'}, location='main')

    if authentication_status:
        authenticator.logout('Logout', 'sidebar')
        st.sidebar.write(f'Welcome, *{name}*')

        # Initialize the agent session id if not already set
        if st.session_state.session_id is None:
            st.session_state.session_id = session_generator()

        # Sidebar for conversation history and controls
        st.sidebar.title("Conversation Controls")
        if st.sidebar.button("Toggle History"):
            st.session_state.show_history = not st.session_state.show_history
        if st.sidebar.button("Clear History"):
            st.session_state.conversation = []
            st.session_state.session_id = session_generator()

        # Main chat interface
        chat_container = st.container()
        with chat_container:
            for interaction in st.session_state.conversation:
                if 'user' in interaction:
                    st.markdown(f'<div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;"><span style="color: #4A90E2; font-weight: bold;">You:</span> {interaction["user"]}</div>', unsafe_allow_html=True)
                elif 'assistant' in interaction:
                    st.markdown(f'<div style="background-color: #e6f3ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;"><span style="color: #50E3C2; font-weight: bold;">Assistant:</span> {interaction["assistant"]}</div>', unsafe_allow_html=True)

        # User input area
        user_input = st.text_input("Ask a question:", key="user_input", on_change=submit_question)
        col1, col2, col3 = st.columns([1, 1, 5])
        with col1:
            submit_button = st.button("Submit", on_click=submit_question)
        with col2:
            clear_button = st.button("Clear Input", on_click=clear_input)

def clear_input():
    st.session_state.user_input = ""
    st.experimental_rerun()

    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')

if __name__ == '__main__':
    main()
                    {'AttributeName': 'id', 'KeyType': 'HASH'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'id', 'AttributeType': 'S'}
                ],
                ProvisionedThroughput={
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            )
            # Wait for the table to be created
            table.meta.client.get_waiter('table_exists').wait(TableName=table_name)
            print(f"Table {table_name} created successfully.")
        else:
            print(f"Table {table_name} already exists.")
    except ClientError as e:
        print(f"Error ensuring DynamoDB table exists: {e}")
