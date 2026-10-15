
from minieval import evaluate

# Replace this example with your own LLM.
def my_llm(question: str) -> str:
    return "Your LLM's generated answer goes here."

report = evaluate(
    model=my_llm,
    judge_model="llama3.2:latest",
    threshold=0.7,
)

report.print_summary()
report.print_results()

json_file, csv_file = report.save("evaluation_results")
print("Saved:", json_file, csv_file)
