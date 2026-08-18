"""A tiny course-glossary lookup tool.

Run it from the command line:

    python glossary.py token

Or import look_up() into your own code. It's deliberately small — this is
the project you'll practice using Claude Code on.
"""

import sys

# The course glossary — the key terms from the course.
GLOSSARY = {
    "model": "Learned system mapping inputs to outputs.",
    "parameter": "Learned numerical weight inside model.",
    "token": "Basic text unit; word, subword, punctuation.",
    "tokenizer": "Converts text ↔ tokens.",
    "context window": "Maximum tokens model can process at once.",
    "prompt": "Instructions/context given to model.",
    "pretraining": "Broad training on large datasets.",
    "post-training": "Alignment and capability refinement after pretraining.",
    "fine-tuning": "Additional training for specific behavior/domain.",
    "rlhf": "Training from human preference feedback.",
    "probability distribution": "Likelihood assigned to each possible next token.",
    "temperature": "Randomness control; lower = more predictable.",
    "max output tokens": "Generation-length limit.",
    "transformer": "Dominant LLM architecture.",
    "attention": "Mechanism relating tokens to relevant tokens.",
    "self-attention": "Attention among tokens in the same sequence.",
    "embedding": "Vector representation of text or other data.",
    "layer": "Repeated processing block in a neural network.",
    "multimodal model": "Processes text plus images, audio, or video.",
    "hallucination": "Plausible but unsupported or false output.",
    "rag": "Retrieval-Augmented Generation: retrieve documents, then generate using them.",
    "chunking": "Splitting documents into retrievable pieces.",
    "tool calling": "Model requests external functions or APIs.",
    "agent": "Model-driven system that plans, uses tools, and acts.",
    "memory": "Retained information across interactions.",
    "guardrail": "Rule or mechanism limiting unsafe/invalid behavior.",
    "latency": "Time required to produce a response.",
    "quantization": "Lower-precision weights for faster, cheaper inference.",
    "distillation": "Training a smaller model from a stronger model.",
    "open-weight model": "Downloadable model weights; not necessarily open-source.",
}


def look_up(term):
    """Return the definition for `term`, or a friendly not-found message."""
    key = term.lower()
    if key in GLOSSARY:
        return GLOSSARY[key]
    available = ", ".join(sorted(GLOSSARY))
    return f'No entry for "{term}". Try one of: {available}.'


def main(argv):
    if len(argv) < 2:
        print("Usage: python glossary.py <term>")
        return 1
    term = " ".join(argv[1:])
    print(look_up(term))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
