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
from botocore.exceptions import ClientError

# Load environment variables
load_dotenv()

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
        if e.response['Error']['Code'] == 'AccessDeniedException':
            logging.error(f"AccessDeniedException: {e}")
            st.warning("Unable to save conversation due to permissions. Your feedback is still valuable!")
        else:
            logging.error(f"Unexpected error when saving to DynamoDB: {e}")
            st.warning("Unable to save conversation. Your feedback is still valuable!")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        st.warning("An unexpected error occurred. Your feedback is still valuable!")

def main():
    st.title("Conversational AI - Plant Technician")

    # Authentication
    name, authentication_status, username = authenticator.login(fields={'form_name': 'Login'}, location='main')

    if authentication_status:
        authenticator.logout('Logout', 'main')
        st.write(f'Welcome *{name}*')

        # Initialize the agent session id if not already set
        if st.session_state.session_id is None:
            st.session_state.session_id = session_generator()

        # Taking user input
        user_prompt = st.text_input("Message:")

        if user_prompt:
            try:
                # Add the user's prompt to the conversation state
                st.session_state.conversation.append({'user': user_prompt})

                # Format and add the answer to the conversation state
                response = bedrock_client.invoke_agent(
                    agentId=BEDROCK_AGENT_ID,
                    agentAliasId=BEDROCK_AGENT_ALIAS,
                    sessionId=st.session_state.session_id,
                    endSession=False,
                    inputText=user_prompt
                )
                results = response.get("completion")
                answer = ""
                for stream in results:
                    answer += process_stream(stream)
                st.session_state.conversation.append(
                    {'assistant': answer})

                # Save the conversation to DynamoDB
                save_to_dynamodb(username, st.session_state.session_id, st.session_state.conversation)

            except Exception as e:
                # Display an error message if an exception occurs
                st.error("An error occurred. Please try again later.")
                print(f"ERROR: Exception when calling Bedrock Agent: {e}")

        # Display the conversation
        for interaction in st.session_state.conversation:
            if 'user' in interaction:
                st.markdown(f'<div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;"><span style="color: #4A90E2; font-weight: bold;">User:</span> {interaction["user"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="background-color: #e6f3ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;"><span style="color: #50E3C2; font-weight: bold;">Assistant:</span> {interaction["assistant"]}</div>', unsafe_allow_html=True)

        # Add feedback buttons
        if st.button("👍 Helpful"):
            save_to_dynamodb(username, st.session_state.session_id, st.session_state.conversation + [{"feedback": "helpful"}])
            st.success("Thank you for your feedback!")

        if st.button("👎 Not Helpful"):
            save_to_dynamodb(username, st.session_state.session_id, st.session_state.conversation + [{"feedback": "not helpful"}])
            st.success("Thank you for your feedback!")

        st.info("Note: If you see a warning about saving the conversation, your feedback is still valuable and has been recorded.")

    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')

if __name__ == '__main__':
    main()
