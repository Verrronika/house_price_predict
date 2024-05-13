import numpy as np

def calculate_psi(expected, actual, bins=10):
    expected = np.array(expected)
    actual = np.array(actual)

    # классификация данных в диапазоны
    breaks = np.linspace(0, 1, bins + 1)
    expected_counts = np.histogram(expected, breaks)[0]
    actual_counts = np.histogram(actual, breaks)[0]
    
    # замена нулевых значений на очень маленький (для предотвращения деления на 0)
    expected_counts = np.where(expected_counts == 0, 0.0001, expected_counts)
    actual_counts = np.where(actual_counts == 0, 0.0001, actual_counts)

    # psi
    return np.sum((actual_counts / len(actual) - expected_counts / len(expected)) * np.log((actual_counts / len(actual)) / (expected_counts / len(expected))))