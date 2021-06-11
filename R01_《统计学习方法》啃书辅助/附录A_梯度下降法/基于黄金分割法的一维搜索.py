from code.gradient_descent import golden_section_for_line_search


if __name__ == "__main__":
    print(golden_section_for_line_search(lambda x: x ** 2, -10, 5, epsilon=1e-6))  # 5.263005013597177e-06
