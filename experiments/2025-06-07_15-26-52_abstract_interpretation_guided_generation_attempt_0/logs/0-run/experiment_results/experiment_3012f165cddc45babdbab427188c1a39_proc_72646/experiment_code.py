import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import ast, numpy as np, matplotlib.pyplot as plt

# Define synthetic tasks
tasks = [
    {
        "name": "add",
        "prompt": "Add two numbers",
        "testcases": [((1, 2), 3), ((-1, 5), 4)],
    },
    {
        "name": "divide",
        "prompt": "Divide two numbers",
        "testcases": [((6, 3), 2), ((5, 0), None)],
    },
    {
        "name": "factorial",
        "prompt": "Compute factorial",
        "testcases": [((0,), 1), ((5,), 120)],
    },
]


def generate_code_naive(task):
    if task["name"] == "add":
        return "def add(a,b):\n    return a+b"
    if task["name"] == "divide":
        return "def divide(a,b):\n    return a/b"
    if task["name"] == "factorial":
        return "def factorial(n):\n    res=1\n    for i in range(1,n+1):\n        res*=i\n    return res"
    return ""


def refine_code(code, task):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            # inject zero-check before division
            lines = code.splitlines()
            for i, ln in enumerate(lines):
                if "return" in ln and "/" in ln:
                    indent = ln[: ln.index("return")]
                    expr = ln.strip().replace("return ", "")
                    var = expr.split("/")[-1]
                    lines[i] = (
                        f"{indent}if {var}==0: return None\n{indent}else: return {expr}"
                    )
            return "\n".join(lines)
    return code


def evaluate_code(code, task):
    local = {}
    try:
        exec(code, {}, local)
        fn = local[task["name"]]
        for args, expected in task["testcases"]:
            out = fn(*args)
            if out != expected:
                return False
        return True
    except Exception:
        return False


baseline_results = []
guided_results = []
generated = []

for task in tasks:
    code0 = generate_code_naive(task)
    ok0 = evaluate_code(code0, task)
    baseline_results.append(ok0)
    code1 = refine_code(code0, task)
    ok1 = evaluate_code(code1, task)
    guided_results.append(ok1)
    generated.append({"task": task["name"], "baseline": code0, "guided": code1})

# Compute metrics
rate0 = sum(baseline_results) / len(tasks)
rate1 = sum(guided_results) / len(tasks)
print(f"Baseline Error-Free Rate: {rate0:.2f}")
print(f"Guided   Error-Free Rate: {rate1:.2f}")

# Save experiment data
experiment_data = {
    "metrics": {"baseline": baseline_results, "guided": guided_results},
    "generated_code": generated,
}
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# Visualization
plt.bar(["baseline", "guided"], [rate0, rate1], color=["red", "green"])
plt.ylabel("Error-Free Generation Rate")
plt.title("AIGG vs Baseline")
plt.savefig(os.path.join(working_dir, "error_rates.png"))
