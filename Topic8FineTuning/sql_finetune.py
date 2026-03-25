"""
Topic 8: Fine-Tuning Exercise — Text-to-SQL with Tinker
========================================================
Steps:
  1. Load dataset, split into train / test
  2. Evaluate BASE model accuracy on 200 test questions
  3. Tokenize training data with loss-weight masks
  4. Run supervised fine-tuning loop (1 epoch, batch size 256)
  5. Evaluate FINE-TUNED model accuracy on same 200 test questions
  6. Test on 5 novel out-of-distribution schema questions

Usage:
  export TINKER_API_KEY=<your_key>
  python sql_finetune.py

Dependencies:
  pip install tinker transformers python-dotenv
"""

import json
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

import tinker
from tinker import types

from sql_matches import sql_matches  # execution-based SQL comparison

# ──────────────────────────────────────────────
# 0. CONFIG
# ──────────────────────────────────────────────
DATA_FILE        = "sql_create_context_v4.json"
BASE_MODEL       = "meta-llama/Llama-3.2-1B"
LORA_RANK        = 32
NUM_TEST         = 200
BATCH_SIZE       = 256
LEARNING_RATE    = 5e-4
NUM_EPOCHS       = 1
NUM_EVAL_WORKERS = 8   # parallel threads for evaluation sampling
RANDOM_SEED      = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ──────────────────────────────────────────────
# 1. LOAD & SPLIT DATA
# ──────────────────────────────────────────────
print("Loading dataset...")
with open(DATA_FILE) as f:
    data = json.load(f)

random.shuffle(data)
test_data  = data[:NUM_TEST]
train_data = data[NUM_TEST:]
print(f"Train: {len(train_data)} examples  |  Test: {len(test_data)} examples")

# ──────────────────────────────────────────────
# 2. TINKER CLIENT & TOKENIZER
# ──────────────────────────────────────────────
print("\nCreating Tinker training client...")
service_client  = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(
    base_model=BASE_MODEL,
    rank=LORA_RANK,
)
tokenizer = training_client.get_tokenizer()
print(f"Model: {BASE_MODEL}  |  LoRA rank: {LORA_RANK}")

# Helper: convert tinker array types to numpy
to_arr = lambda x: x.to_numpy() if hasattr(x, 'to_numpy') else np.array(x.tolist())

# ──────────────────────────────────────────────
# 3. HELPER: PROMPT FORMAT
# ──────────────────────────────────────────────
def format_prompt(example: dict) -> tuple[str, str]:
    """Return (prompt_str, completion_str) for one example."""
    prompt = (
        f"Table schema:\n"
        f"{example['context']}\n"
        f"Question: {example['question']}\n"
        f"SQL: "
    )
    completion = example["answer"]
    return prompt, completion

# ──────────────────────────────────────────────
# 4. HELPER: TOKENIZE → Datum
# ──────────────────────────────────────────────
def process_example(example: dict) -> types.Datum:
    """Tokenize one example with loss-weight mask (0 on prompt, 1 on completion)."""
    prompt, completion = format_prompt(example)

    prompt_tokens      = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights     = [0.0] * len(prompt_tokens)

    completion_str     = f" {completion}\n\n"
    completion_tokens  = tokenizer.encode(completion_str, add_special_tokens=False)
    completion_weights = [1.0] * len(completion_tokens)

    tokens  = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    # Shift by 1 for next-token prediction
    input_tokens  = tokens[:-1]
    target_tokens = tokens[1:]
    weights       = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "target_tokens": np.array(target_tokens, dtype=np.int64),
            "weights":       np.array(weights,       dtype=np.float32),
        },
    )

# ──────────────────────────────────────────────
# 5. HELPER: SAMPLE ONE QUESTION
# ──────────────────────────────────────────────
def sample_sql(sampling_client, context: str, question: str) -> str:
    """Generate SQL from the model given schema + question."""
    prompt = f"Table schema:\n{context}\nQuestion: {question}\nSQL: "
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    model_input   = types.ModelInput.from_ints(tokens=prompt_tokens)
    params        = types.SamplingParams(
        max_tokens=150, temperature=0.0, stop=["\n\n", "Question:"]
    )
    result = sampling_client.sample(
        prompt=model_input, sampling_params=params, num_samples=1
    ).result()
    if result.sequences:
        return tokenizer.decode(result.sequences[0].tokens).strip()
    return ""

# ──────────────────────────────────────────────
# 6. HELPER: EVALUATE TEST SET
# ──────────────────────────────────────────────
def evaluate_test_set(sampling_client, test_data: list, label: str) -> float:
    """Compute accuracy over test_data using execution-based SQL comparison."""
    print(f"\n--- Evaluating {label} model on {len(test_data)} test questions ---")
    correct = 0

    def eval_one(ex):
        generated = sample_sql(sampling_client, ex["context"], ex["question"])
        return sql_matches(generated, ex["answer"], schema=ex["context"])

    with ThreadPoolExecutor(max_workers=NUM_EVAL_WORKERS) as executor:
        futures = {executor.submit(eval_one, ex): i for i, ex in enumerate(test_data)}
        for i, future in enumerate(as_completed(futures)):
            if future.result():
                correct += 1
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(test_data)} evaluated  (correct so far: {correct})")

    accuracy = correct / len(test_data)
    print(f"{label} accuracy: {accuracy:.2%}  ({correct}/{len(test_data)})")
    return accuracy

# ──────────────────────────────────────────────
# 7. EVALUATE BASE MODEL
# ──────────────────────────────────────────────
print("\nSaving base weights to get sampling client...")
base_sampler = training_client.save_weights_and_get_sampling_client(name="base-model")
base_accuracy = evaluate_test_set(base_sampler, test_data, "Base")

# ──────────────────────────────────────────────
# 8. PREPARE TRAINING DATA
# ──────────────────────────────────────────────
print(f"\nTokenizing {len(train_data)} training examples...")
processed_train = [process_example(ex) for ex in train_data]
random.shuffle(processed_train)
print("Tokenization complete.")

# ──────────────────────────────────────────────
# 9. TRAINING LOOP
# ──────────────────────────────────────────────
print(f"\n=== Starting Fine-Tuning ===")
print(f"Epochs: {NUM_EPOCHS}  |  Batch size: {BATCH_SIZE}  |  LR: {LEARNING_RATE}")

total_batches = (len(processed_train) + BATCH_SIZE - 1) // BATCH_SIZE
print(f"Batches per epoch: {total_batches}")

step = 0
for epoch in range(NUM_EPOCHS):
    random.shuffle(processed_train)
    for batch_start in range(0, len(processed_train), BATCH_SIZE):
        batch = processed_train[batch_start : batch_start + BATCH_SIZE]
        if not batch:
            break

        # Submit forward+backward and optimizer step (pipelined)
        fwdbwd_future = training_client.forward_backward(batch, "cross_entropy")
        optim_future  = training_client.optim_step(
            types.AdamParams(learning_rate=LEARNING_RATE)
        )

        fwdbwd_result = fwdbwd_future.result()
        optim_future.result()  # wait for optimizer step to complete

        # Compute weighted cross-entropy loss over completion tokens only
        logprobs = np.concatenate([to_arr(o["logprobs"]) for o in fwdbwd_result.loss_fn_outputs])
        weights  = np.concatenate([to_arr(d.loss_fn_inputs["weights"]) for d in batch])
        loss     = float(-np.dot(logprobs, weights) / (weights.sum() + 1e-8))

        step += 1
        batch_num = batch_start // BATCH_SIZE + 1
        if step % 10 == 0 or batch_num == total_batches:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}  batch {batch_num}/{total_batches}  "
                  f"step {step}  loss: {loss:.4f}")

print("\nTraining complete!")

# ──────────────────────────────────────────────
# 10. EVALUATE FINE-TUNED MODEL
# ──────────────────────────────────────────────
print("\nSaving fine-tuned weights...")
ft_sampler = training_client.save_weights_and_get_sampling_client(name="sql-finetuned")
ft_accuracy = evaluate_test_set(ft_sampler, test_data, "Fine-tuned")

print(f"\n{'='*50}")
print(f"RESULTS SUMMARY")
print(f"{'='*50}")
print(f"Base model accuracy:        {base_accuracy:.2%}")
print(f"Fine-tuned model accuracy:  {ft_accuracy:.2%}")
print(f"Improvement:                +{ft_accuracy - base_accuracy:.2%}")

# ──────────────────────────────────────────────
# 11. NOVEL OUT-OF-DISTRIBUTION QUESTIONS
# ──────────────────────────────────────────────
novel_questions = [
    # Easy
    {
        "label": "Easy-1: employees WHERE",
        "context": "CREATE TABLE employees (id INTEGER, name VARCHAR, salary REAL, department VARCHAR)",
        "question": "What are the names of employees in the engineering department?",
        "expected": "SELECT name FROM employees WHERE department = 'engineering'",
    },
    {
        "label": "Easy-2: products COUNT",
        "context": "CREATE TABLE products (id INTEGER, name VARCHAR, price REAL, category VARCHAR)",
        "question": "How many products cost more than 50 dollars?",
        "expected": "SELECT COUNT(*) FROM products WHERE price > 50",
    },
    # Medium
    {
        "label": "Medium-1: students MAX",
        "context": "CREATE TABLE students (id INTEGER, name VARCHAR, score INTEGER, class VARCHAR)",
        "question": "What is the highest score in the science class?",
        "expected": "SELECT MAX(score) FROM students WHERE class = 'science'",
    },
    {
        "label": "Medium-2: orders TOP-3",
        "context": "CREATE TABLE orders (id INTEGER, customer VARCHAR, amount REAL, date VARCHAR)",
        "question": "List the top 3 customers by total order amount.",
        "expected": "SELECT customer, SUM(amount) FROM orders GROUP BY customer ORDER BY SUM(amount) DESC LIMIT 3",
    },
    # Hard
    {
        "label": "Hard-1: JOIN + GROUP BY",
        "context": (
            "CREATE TABLE courses (id INTEGER, name VARCHAR, department VARCHAR); "
            "CREATE TABLE enrollments (student_id INTEGER, course_id INTEGER, grade VARCHAR)"
        ),
        "question": "How many students are enrolled in each department?",
        "expected": "SELECT c.department, COUNT(e.student_id) FROM courses c JOIN enrollments e ON c.id = e.course_id GROUP BY c.department",
    },
]

print(f"\n{'='*50}")
print("NOVEL SCHEMA QUESTIONS (out-of-distribution)")
print(f"{'='*50}")

for q in novel_questions:
    generated = sample_sql(ft_sampler, q["context"], q["question"])
    match     = sql_matches(generated, q["expected"], schema=q["context"])
    status    = "✓ CORRECT" if match else "✗ WRONG"
    print(f"\n[{q['label']}]  {status}")
    print(f"  Question:  {q['question']}")
    print(f"  Expected:  {q['expected']}")
    print(f"  Generated: {generated}")

print("\nDone! Save this output to a text file for your portfolio.")