import pytest

from calculator.calc import (
    add,
    average,
    divide,
    factorial,
    fibonacci,
    is_palindrome,
    is_prime,
    max_value,
    modulo,
    square,
    subtract,
)


def test_add():
    assert add(2, 3) == 5


def test_subtract():
    assert subtract(5, 3) == 2


def test_modulo():
    assert modulo(10, 3) == 1
    assert modulo(7, 2) == 1


def test_square():
    assert square(4) == 16
    assert square(0) == 0
    assert square(-3) == 9


def test_max_value():
    assert max_value([1, 5, 3]) == 5
    assert max_value([-5, -2, -8]) == -2


def test_is_palindrome():
    assert is_palindrome("racecar") is True
    assert is_palindrome("Racecar") is True
    assert is_palindrome("hello") is False


def test_fibonacci():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(5) == 5


def test_divide():
    assert divide(6, 3) == 2


def test_divide_by_zero():
    with pytest.raises(ValueError):
        divide(5, 0)


def test_average():
    assert average([2, 4, 6]) == 4
    assert average([1, 2, 3, 4]) == 2.5


def test_average_empty_list():
    with pytest.raises(ValueError):
        average([])


def test_is_prime():
    assert is_prime(2) is True
    assert is_prime(7) is True
    assert is_prime(4) is False
    assert is_prime(9) is False
    assert is_prime(15) is False


def test_factorial():
    assert factorial(5) == 120
    assert factorial(3) == 6


def test_factorial_negative():
    with pytest.raises(ValueError):
        factorial(-1)
