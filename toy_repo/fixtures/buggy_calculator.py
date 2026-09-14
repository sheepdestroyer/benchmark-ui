ALLOWED_OPS = {"+", "-", "*", "/"}


def parse_and_eval(expression: str) -> float:
    """
    Parses and evaluates a simple math expression sequentially left-to-right.
    Lacks operator precedence (order-of-operations bug).
    Example: parse_and_eval("3 + 4 * 2") -> 14.0 instead of 11.0
    """
    if not isinstance(expression, str):
        raise TypeError("Expression must be a string")

    tokens = expression.split()
    if not tokens:
        return 0.0

    if len(tokens) % 2 == 0:
        raise ValueError(f"Invalid expression syntax: '{expression}'")

    for i, token in enumerate(tokens):
        if i % 2 == 1 and token not in ALLOWED_OPS:
            raise ValueError(f"Unsupported operator: '{token}'")
    nums = [float(tokens[i]) for i in range(0, len(tokens), 2)]
    ops = [tokens[i] for i in range(1, len(tokens), 2)]

    if not set(ops).issubset(ALLOWED_OPS):
        for op in ops:
            if op not in ALLOWED_OPS:
                raise ValueError(f"Unsupported operator: {op}")

    # Buggy evaluation: left-to-right sequentially without operator precedence
    result = nums[0]
    for i, op in enumerate(ops):
        next_num = nums[i + 1]
        if op == "+":
            result += next_num
        elif op == "-":
            result -= next_num
        elif op == "*":
            result *= next_num
        elif op == "/":
            if next_num == 0:
                raise ZeroDivisionError("division by zero")
            result /= next_num

    return result
