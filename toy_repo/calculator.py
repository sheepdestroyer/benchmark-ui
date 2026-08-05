def parse_and_eval(expression):
    """
    Parses and evaluates a simple math expression.
    Supported operations: +, -, *, /
    Example: parse_and_eval("3 + 4 * 2") -> 11
    """
    tokens = expression.split()
    if not tokens:
        return 0

    nums = [float(tokens[i]) for i in range(0, len(tokens), 2)]
    ops = [tokens[i] for i in range(1, len(tokens), 2)]

    for op in ops:
        if op not in ("+", "-", "*", "/"):
            raise ValueError(f"Unsupported operator: {op}")

    # Handle multiplication and division first
    i = 0
    while i < len(ops):
        if ops[i] == '*':
            nums[i] *= nums[i+1]
            nums.pop(i+1)
            ops.pop(i)
        elif ops[i] == '/':
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
        if ops[i] == '+':
            nums[i] += nums[i+1]
            nums.pop(i+1)
            ops.pop(i)
        elif ops[i] == '-':
            nums[i] -= nums[i+1]
            nums.pop(i+1)
            ops.pop(i)
        else:
            i += 1

    return nums[0]
