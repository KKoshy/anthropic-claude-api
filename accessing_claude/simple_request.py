"""This is an example for placing a simple request to Claude"""

from anthropic import Anthropic
from dotenv import load_dotenv
from log.logger import get_logger

load_dotenv()
log = get_logger(__name__)

# Creating API client
client = Anthropic()
model = "claude-haiku-4-5"
log.info("Placing a simple request to Claude Haiku Model")
ssd_message = client.messages.create(
    model=model,
    max_tokens=1000,
    messages=[
        {
            'role': 'user',
            'content': 'What is SSD? Answer in one sentence'
        }
    ]
)
log.info(ssd_message)
log.info("Accessing only the text message returned")
log.info(ssd_message.content[0].text)
