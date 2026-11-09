import os

from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
prompt = "Explain why this distributed job can run twice and suggest a fix."

decision = client.decisions.create(
    model="gpt-6-luna",
    input=prompt,
    questions=[
        {
            "type": "choice",
            "name": "complexity",
            "instructions": "Classify the work required to answer this request.",
            "choices": [
                {
                    "value": "simple",
                    "description": "A short rewrite or factual question.",
                },
                {
                    "value": "complex",
                    "description": "Multi-step reasoning or debugging.",
                },
            ],
        }
    ],
)

answer = decision.answers[0] if decision.answers else None
if answer is not None and answer.type == "refusal":
    raise RuntimeError("Routing refused; handle the request separately.")

print("\nRouting decision")
print(f"  Question:   {answer.name}")
print(f"  Choice:     {answer.choice}")
print(f"  Confidence: {answer.confidence:.0%}")
print("  Probabilities:")
for option in answer.probabilities:
    print(f"    {option.value:<8} {option.probability:.0%}")

# Illustrative threshold: tune using labeled application requests.
use_small_model = (
    answer is not None
    and answer.type == "choice"
    and answer.choice == "simple"
    and answer.confidence >= 0.8
)
model = "gpt-6-luna" if use_small_model else "gpt-6.1-sol"

response = client.responses.create(model=model, input=prompt)
print("\nFinal response")
print(f"  Model: {model}")
print(f"  Answer:\n{response.output_text}")

