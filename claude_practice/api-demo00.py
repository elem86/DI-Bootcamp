import sys
from dotenv import load_dotenv
import anthropic

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

client = anthropic.Anthropic()
MODEL = "claude-sonnet-5"
PROMPT = "Give a catchy 5-word slogan for a coffee shop. Make it fun and memorable."


message = client.messages.create(
    model=MODEL,
    max_tokens=500,
    thinking={"type": "adaptive"},
    output_config={"effort": "high"},
    messages=[{"role": "user", "content": PROMPT}],
)

text = next(block.text for block in message.content if block.type == "text")
print(text)
