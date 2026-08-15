import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"


def load_model():
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Loading model...")

    llm_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype="auto",
        device_map="auto",
    )

    llm_model.eval()

    print("Model loaded successfully.")

    return tokenizer, llm_model


def generate_answer(
    tokenizer,
    model,
    question,
    context,
):
    prompt = f"""
You are Dunder Mifflin's internal employee policy assistant.

Answer the employee's question using ONLY the provided context.

Rules:
- Use only information explicitly contained in the context.
- Do not use outside knowledge.
- Do not invent or assume company policies.
- If the context does not contain enough information to answer the question, say:
  "I don't have enough information in the provided company policies to answer that."
- Keep the answer concise and direct.

Context:
{context}

Employee question:
{question}

Answer:
"""

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
        )

    generated_tokens = outputs[
        0
    ][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return answer.strip()


if __name__ == "__main__":

    tokenizer, llm_model = load_model()

    test_context = """
Dunder Mifflin Paper Company
Leave Policy

Employees receive 18 days of paid annual leave per calendar year.
Unused annual leave may be carried forward into the following
calendar year, up to a maximum of 5 days.
"""

    test_question_1 = (
        "How many annual leave days can an employee carry forward?"
    )

    answer = generate_answer(
        tokenizer=tokenizer,
        model=llm_model,
        question=test_question_1,
        context=test_context,
    )

    print("\n" + "=" * 70)
    print("TEST QUESTION")
    print("=" * 70)
    print(test_question_1)

    print("\n" + "=" * 70)
    print("GENERATED ANSWER")
    print("=" * 70)
    print(answer)


    test_question_2 = (
            "What is Dunder Mifflin's policy for employees bringing pet llamas into the office?"
        )
    
    answer = generate_answer(
            tokenizer=tokenizer,
            model=llm_model,
            question=test_question_2,
            context=test_context,
        )
    
    print("\n" + "=" * 70)
    print("TEST QUESTION")
    print("=" * 70)
    print(test_question_2)
    
    print("\n" + "=" * 70)
    print("GENERATED ANSWER")
    print("=" * 70)
    print(answer)