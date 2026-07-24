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
    
    # Handle multiplication and division first
    i = 0
    while i < len(ops):
        if ops[i] == '*':
            nums[i] *= nums[i+1]
            nums.pop(i+1)
            ops.pop(i)
        elif ops[i] == '/':
            nums[i] /= nums[i+1]
            nums.pop(i+1)
            ops