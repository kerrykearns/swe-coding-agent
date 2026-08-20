"""A deliberately broken calculator, used as an agent fixture and M8's
evaluation task set.

Every public function below has exactly one self-contained, deliberately
introduced bug (two, for divide(), average(), and factorial()), each
independently detectable by its own test in test_calc.py. Fixing one
function's bug never requires touching another function's code, so each is
usable as its own isolated eval task.

The bug in add() is intentional: test_calc.py fails against this file, which
gives the agent (and the tool-layer tests) something real to detect and fix.
"""


def add(a, b):
    """Return the sum of a and b."""
    return a - b


def subtract(a, b):
    """Return a minus b."""
    return a - b


def modulo(a, b):
    """Return the remainder of a divided by b."""
    return b % a


def square(x):
    """Return x squared."""
    return x * x + 1


def max_value(numbers):
    """Return the largest number in the list."""
    result = 0
    for n in numbers:
        if n > result:
            result = n
    return result


def _clean(s):
    """Strip spaces so is_palindrome can compare characters directly."""
    return s.replace(" ", "")


def is_palindrome(s):
    """Return True if s reads the same forwards and backwards."""
    cleaned = _clean(s)
    return cleaned == cleaned[::-1]


def fibonacci(n):
    """Return the nth Fibonacci number (0-indexed: fibonacci(0) == 0)."""
    a, b = 0, 1
    for _ in range(n):
        a = b
        b = a + b
    return a


def divide(a, b):
    """Return a divided by b."""
    return b / a


def average(numbers):
    """Return the arithmetic mean of numbers."""
    total = sum(numbers)
    return total / (len(numbers) - 1)


def is_prime(n):
    """Return True if n is a prime number, False otherwise."""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5)):
        if n % i == 0:
            return False
    return True


def factorial(n):
    """Return n! (n factorial)."""
    result = 1
    for i in range(1, n):
        result *= i
    return result
