def parse_and_eval(expression: str) -> float:
    """
    Parses and evaluates a simple math expression.
    Supported operations: +, -, *, /
    Example: parse_and_eval("3 + 4 * 2") -> 11.0
    """
    if not isinstance(expression, str):
        raise TypeError("Expression must be a string")

    tokens = expression.split()
    if not tokens:
        return 0.0

    if len(tokens) % 2 == 0:
        raise ValueError(f"Invalid expression syntax: '{expression}'")

    allowed_ops = {"+", "-", "*", "/"}
    for i, token in enumerate(tokens):
        if i % 2 == 1 and token not in allowed_ops:
            raise ValueError(f"Unsupported operator: '{token}'")
    nums = [float(tokens[i]) for i in range(0, len(tokens), 2)]
    ops = [tokens[i] for i in range(1, len(tokens), 2)]

    if not set(ops).issubset(allowed_ops):
        for op in ops:
            if op not in allowed_ops:
                raise ValueError(f"Unsupported operator: {op}")

    # Handle multiplication and division first
    i = 0
    while i < len(ops):
        if ops[i] == "*":
            nums[i] *= nums[i+1]
            nums.pop(i+1)
            ops.pop(i)
        elif ops[i] == "/":
            if nums[i+1] == 0:
                raise ZeroDivisionError("division by zero")
            nums[i] /= nums[i+1]
            nums.pop(i+1)
            ops.pop(i)
        else:
            i += 1

    # Handle addition and subtraction
    i = 0
    while i < len(ops):
        if ops[i] == "+":
            nums[i] += nums[i+1]
            nums.pop(i+1)
            ops.pop(i)
        elif ops[i] == "-":
            nums[i] -= nums[i+1]
            nums.pop(i+1)
            ops.pop(i)
        else:
            i += 1

    return nums[0]
