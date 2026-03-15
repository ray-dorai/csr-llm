"""Problem bank and execution sandbox for code self-play.

Problems are organized in 5 levels of difficulty. The executor runs
model-generated code in a subprocess with a timeout for safety.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass


@dataclass
class CodeProblem:
    name: str
    level: int       # 1–5
    prompt: str      # function signature + docstring (the generation prefix)
    solution: str    # reference solution body (indented, for random baseline)
    tests: str       # assert statements to verify correctness


PROBLEMS: list[CodeProblem] = [

    # ── Level 1: single-operation ──────────────────────────────────────────
    CodeProblem(
        name="add",
        level=1,
        prompt='def add(a: int, b: int) -> int:\n    """Return the sum of a and b."""\n',
        solution='    return a + b\n',
        tests=(
            'assert add(1, 2) == 3\n'
            'assert add(0, 0) == 0\n'
            'assert add(-1, 1) == 0\n'
            'assert add(100, 200) == 300\n'
        ),
    ),
    CodeProblem(
        name="multiply",
        level=1,
        prompt='def multiply(a: int, b: int) -> int:\n    """Return the product of a and b."""\n',
        solution='    return a * b\n',
        tests=(
            'assert multiply(3, 4) == 12\n'
            'assert multiply(0, 5) == 0\n'
            'assert multiply(-2, 3) == -6\n'
        ),
    ),
    CodeProblem(
        name="is_even",
        level=1,
        prompt='def is_even(n: int) -> bool:\n    """Return True if n is even, False otherwise."""\n',
        solution='    return n % 2 == 0\n',
        tests=(
            'assert is_even(4) == True\n'
            'assert is_even(3) == False\n'
            'assert is_even(0) == True\n'
        ),
    ),
    CodeProblem(
        name="max_of_two",
        level=1,
        prompt='def max_of_two(a: int, b: int) -> int:\n    """Return the larger of a and b."""\n',
        solution='    return a if a > b else b\n',
        tests=(
            'assert max_of_two(3, 5) == 5\n'
            'assert max_of_two(7, 2) == 7\n'
            'assert max_of_two(4, 4) == 4\n'
        ),
    ),
    CodeProblem(
        name="absolute_value",
        level=1,
        prompt='def absolute_value(n: int) -> int:\n    """Return the absolute value of n."""\n',
        solution='    return n if n >= 0 else -n\n',
        tests=(
            'assert absolute_value(5) == 5\n'
            'assert absolute_value(-3) == 3\n'
            'assert absolute_value(0) == 0\n'
        ),
    ),

    # ── Level 2: simple algorithms ─────────────────────────────────────────
    CodeProblem(
        name="factorial",
        level=2,
        prompt='def factorial(n: int) -> int:\n    """Return n factorial. Assume n >= 0."""\n',
        solution='    if n == 0:\n        return 1\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result\n',
        tests=(
            'assert factorial(0) == 1\n'
            'assert factorial(1) == 1\n'
            'assert factorial(5) == 120\n'
            'assert factorial(3) == 6\n'
        ),
    ),
    CodeProblem(
        name="is_palindrome",
        level=2,
        prompt='def is_palindrome(s: str) -> bool:\n    """Return True if s reads the same forwards and backwards."""\n',
        solution='    return s == s[::-1]\n',
        tests=(
            'assert is_palindrome("racecar") == True\n'
            'assert is_palindrome("hello") == False\n'
            'assert is_palindrome("a") == True\n'
            'assert is_palindrome("") == True\n'
        ),
    ),
    CodeProblem(
        name="sum_list",
        level=2,
        prompt='def sum_list(lst: list) -> int:\n    """Return the sum of all elements in lst."""\n',
        solution='    total = 0\n    for x in lst:\n        total += x\n    return total\n',
        tests=(
            'assert sum_list([1, 2, 3]) == 6\n'
            'assert sum_list([]) == 0\n'
            'assert sum_list([-1, 1]) == 0\n'
            'assert sum_list([10]) == 10\n'
        ),
    ),
    CodeProblem(
        name="count_vowels",
        level=2,
        prompt='def count_vowels(s: str) -> int:\n    """Return the number of vowels (a, e, i, o, u) in s."""\n',
        solution='    return sum(1 for c in s.lower() if c in "aeiou")\n',
        tests=(
            'assert count_vowels("hello") == 2\n'
            'assert count_vowels("rhythm") == 0\n'
            'assert count_vowels("aeiou") == 5\n'
            'assert count_vowels("") == 0\n'
        ),
    ),
    CodeProblem(
        name="fizzbuzz",
        level=2,
        prompt=(
            'def fizzbuzz(n: int) -> str:\n'
            '    """Return "Fizz" if divisible by 3, "Buzz" if by 5, '
            '"FizzBuzz" if both, else str(n)."""\n'
        ),
        solution=(
            '    if n % 15 == 0:\n        return "FizzBuzz"\n'
            '    elif n % 3 == 0:\n        return "Fizz"\n'
            '    elif n % 5 == 0:\n        return "Buzz"\n'
            '    return str(n)\n'
        ),
        tests=(
            'assert fizzbuzz(3) == "Fizz"\n'
            'assert fizzbuzz(5) == "Buzz"\n'
            'assert fizzbuzz(15) == "FizzBuzz"\n'
            'assert fizzbuzz(7) == "7"\n'
            'assert fizzbuzz(1) == "1"\n'
        ),
    ),

    # ── Level 3: moderate complexity ───────────────────────────────────────
    CodeProblem(
        name="is_prime",
        level=3,
        prompt='def is_prime(n: int) -> bool:\n    """Return True if n is a prime number."""\n',
        solution=(
            '    if n < 2:\n        return False\n'
            '    for i in range(2, int(n**0.5) + 1):\n'
            '        if n % i == 0:\n            return False\n'
            '    return True\n'
        ),
        tests=(
            'assert is_prime(2) == True\n'
            'assert is_prime(3) == True\n'
            'assert is_prime(4) == False\n'
            'assert is_prime(17) == True\n'
            'assert is_prime(1) == False\n'
        ),
    ),
    CodeProblem(
        name="reverse_words",
        level=3,
        prompt='def reverse_words(sentence: str) -> str:\n    """Reverse the order of words in sentence."""\n',
        solution='    return " ".join(sentence.split()[::-1])\n',
        tests=(
            'assert reverse_words("hello world") == "world hello"\n'
            'assert reverse_words("one") == "one"\n'
            'assert reverse_words("a b c") == "c b a"\n'
        ),
    ),
    CodeProblem(
        name="count_occurrences",
        level=3,
        prompt='def count_occurrences(lst: list, target) -> int:\n    """Return the number of times target appears in lst."""\n',
        solution='    return sum(1 for x in lst if x == target)\n',
        tests=(
            'assert count_occurrences([1, 2, 2, 3], 2) == 2\n'
            'assert count_occurrences([], 1) == 0\n'
            'assert count_occurrences([1, 1, 1], 1) == 3\n'
            'assert count_occurrences([1, 2, 3], 4) == 0\n'
        ),
    ),
    CodeProblem(
        name="flatten",
        level=3,
        prompt='def flatten(lst: list) -> list:\n    """Return a flat list from a list of lists (one level deep)."""\n',
        solution='    return [x for sublist in lst for x in sublist]\n',
        tests=(
            'assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]\n'
            'assert flatten([[], [1], [2, 3]]) == [1, 2, 3]\n'
            'assert flatten([]) == []\n'
        ),
    ),
    CodeProblem(
        name="binary_search",
        level=3,
        prompt='def binary_search(arr: list, target: int) -> int:\n    """Return index of target in sorted arr, or -1 if not found."""\n',
        solution=(
            '    lo, hi = 0, len(arr) - 1\n'
            '    while lo <= hi:\n'
            '        mid = (lo + hi) // 2\n'
            '        if arr[mid] == target:\n            return mid\n'
            '        elif arr[mid] < target:\n            lo = mid + 1\n'
            '        else:\n            hi = mid - 1\n'
            '    return -1\n'
        ),
        tests=(
            'assert binary_search([1, 3, 5, 7, 9], 5) == 2\n'
            'assert binary_search([1, 3, 5, 7, 9], 1) == 0\n'
            'assert binary_search([1, 3, 5, 7, 9], 6) == -1\n'
            'assert binary_search([], 1) == -1\n'
        ),
    ),

    # ── Level 4: algorithmic ───────────────────────────────────────────────
    CodeProblem(
        name="two_sum",
        level=4,
        prompt='def two_sum(nums: list, target: int) -> list:\n    """Return indices [i, j] where nums[i] + nums[j] == target."""\n',
        solution=(
            '    seen = {}\n'
            '    for i, n in enumerate(nums):\n'
            '        if target - n in seen:\n'
            '            return [seen[target - n], i]\n'
            '        seen[n] = i\n'
            '    return []\n'
        ),
        tests=(
            'assert two_sum([2, 7, 11, 15], 9) == [0, 1]\n'
            'assert two_sum([3, 2, 4], 6) == [1, 2]\n'
            'assert two_sum([3, 3], 6) == [0, 1]\n'
        ),
    ),
    CodeProblem(
        name="valid_parentheses",
        level=4,
        prompt='def valid_parentheses(s: str) -> bool:\n    """Return True if parentheses in s are valid and balanced."""\n',
        solution=(
            '    stack = []\n'
            '    for c in s:\n'
            '        if c == "(":\n            stack.append(c)\n'
            '        elif c == ")":\n'
            '            if not stack:\n                return False\n'
            '            stack.pop()\n'
            '    return len(stack) == 0\n'
        ),
        tests=(
            'assert valid_parentheses("()") == True\n'
            'assert valid_parentheses("(())") == True\n'
            'assert valid_parentheses(")(") == False\n'
            'assert valid_parentheses("(()") == False\n'
            'assert valid_parentheses("") == True\n'
        ),
    ),
    CodeProblem(
        name="merge_sorted",
        level=4,
        prompt='def merge_sorted(a: list, b: list) -> list:\n    """Merge two sorted lists into one sorted list."""\n',
        solution=(
            '    result, i, j = [], 0, 0\n'
            '    while i < len(a) and j < len(b):\n'
            '        if a[i] <= b[j]:\n            result.append(a[i]); i += 1\n'
            '        else:\n            result.append(b[j]); j += 1\n'
            '    return result + a[i:] + b[j:]\n'
        ),
        tests=(
            'assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]\n'
            'assert merge_sorted([], [1, 2]) == [1, 2]\n'
            'assert merge_sorted([1], []) == [1]\n'
            'assert merge_sorted([1, 2], [1, 2]) == [1, 1, 2, 2]\n'
        ),
    ),
    CodeProblem(
        name="rotate_list",
        level=4,
        prompt='def rotate_list(lst: list, k: int) -> list:\n    """Rotate list right by k positions."""\n',
        solution=(
            '    if not lst:\n        return lst\n'
            '    k = k % len(lst)\n'
            '    return lst[-k:] + lst[:-k] if k else lst[:]\n'
        ),
        tests=(
            'assert rotate_list([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3]\n'
            'assert rotate_list([1, 2, 3], 0) == [1, 2, 3]\n'
            'assert rotate_list([1], 5) == [1]\n'
            'assert rotate_list([], 3) == []\n'
        ),
    ),

    # ── Level 5: harder ────────────────────────────────────────────────────
    CodeProblem(
        name="longest_common_prefix",
        level=5,
        prompt='def longest_common_prefix(strs: list) -> str:\n    """Return the longest common prefix among a list of strings."""\n',
        solution=(
            '    if not strs:\n        return ""\n'
            '    prefix = strs[0]\n'
            '    for s in strs[1:]:\n'
            '        while not s.startswith(prefix):\n'
            '            prefix = prefix[:-1]\n'
            '            if not prefix:\n                return ""\n'
            '    return prefix\n'
        ),
        tests=(
            'assert longest_common_prefix(["flower", "flow", "flight"]) == "fl"\n'
            'assert longest_common_prefix(["dog", "racecar"]) == ""\n'
            'assert longest_common_prefix(["abc"]) == "abc"\n'
            'assert longest_common_prefix([]) == ""\n'
        ),
    ),
    CodeProblem(
        name="decode_run_length",
        level=5,
        prompt='def decode_run_length(s: str) -> str:\n    """Decode run-length encoded string. "3a2b" -> "aaabb"."""\n',
        solution=(
            '    result, i = [], 0\n'
            '    while i < len(s):\n'
            '        count = ""\n'
            '        while i < len(s) and s[i].isdigit():\n'
            '            count += s[i]; i += 1\n'
            '        if i < len(s):\n'
            '            result.append(s[i] * int(count or 1)); i += 1\n'
            '    return "".join(result)\n'
        ),
        tests=(
            'assert decode_run_length("3a2b1c") == "aaabbc"\n'
            'assert decode_run_length("1a1b1c") == "abc"\n'
            'assert decode_run_length("5x") == "xxxxx"\n'
        ),
    ),
    CodeProblem(
        name="matrix_transpose",
        level=5,
        prompt='def matrix_transpose(matrix: list) -> list:\n    """Return the transpose of a 2D matrix."""\n',
        solution=(
            '    if not matrix or not matrix[0]:\n        return []\n'
            '    return [[matrix[i][j] for i in range(len(matrix))] '
            'for j in range(len(matrix[0]))]\n'
        ),
        tests=(
            'assert matrix_transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]\n'
            'assert matrix_transpose([[1, 2, 3]]) == [[1], [2], [3]]\n'
            'assert matrix_transpose([]) == []\n'
        ),
    ),
]


def problems_at_level(level: int) -> list[CodeProblem]:
    return [p for p in PROBLEMS if p.level == level]


def execute_solution(
    prompt: str,
    body: str,
    tests: str,
    timeout: int = 5,
) -> tuple[bool, str]:
    """Run prompt+body+tests in a subprocess. Returns (passed, stderr)."""
    # Strip dangerous imports from generated body
    safe_body = _strip_dangerous(body)
    full_code = prompt + safe_body + "\n" + tests
    try:
        result = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "TimeoutExpired"
    except Exception as e:
        return False, str(e)


def _strip_dangerous(code: str) -> str:
    """Remove lines that import os/sys/subprocess to keep execution safe."""
    dangerous = ("import os", "import sys", "import subprocess",
                 "from os", "from sys", "from subprocess", "__import__")
    lines = [ln for ln in code.split("\n") if not any(d in ln for d in dangerous)]
    return "\n".join(lines)


def extract_body(generated: str, prompt: str) -> str:
    """Extract the function body from generated text (everything after prompt)."""
    if generated.startswith(prompt):
        body = generated[len(prompt):]
    else:
        # Model may have regenerated the signature — find the body start
        lines = generated.split("\n")
        body_lines = []
        found_def = False
        for line in lines:
            if line.startswith("def "):
                found_def = True
                continue
            if found_def:
                body_lines.append(line)
        body = "\n".join(body_lines)

    # Stop at the next unindented non-empty line (start of new function/class)
    result_lines = []
    for line in body.split("\n"):
        if line and not line[0].isspace() and result_lines:
            break
        result_lines.append(line)

    return "\n".join(result_lines)
