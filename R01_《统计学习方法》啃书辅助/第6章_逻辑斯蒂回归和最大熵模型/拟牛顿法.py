from code.maximum_entropy_model import bfgs_algorithm_for_maximum_entropy_model

if __name__ == "__main__":
    dataset = [[[1], [1], [1], [1], [2], [2], [2], [2]], [1, 2, 2, 3, 1, 1, 1, 1]]


    def f1(xx, yy):
        return xx == [1] and yy == 1


    def f2(xx, yy):
        return (xx == [1] and yy == 2) or (xx == [1] and yy == 3)


    # {((1,), 1): 0.2500000558794252, ((1,), 2): 0.3749999720602874, ((1,), 3): 0.3749999720602874, ((2,), 1): 0.3333333333333333, ((2,), 2): 0.3333333333333333, ((2,), 3): 0.3333333333333333}
    print(bfgs_algorithm_for_maximum_entropy_model(dataset[0], dataset[1], [f1]))

    # {((1,), 1): 0.24999946967330844, ((1,), 2): 0.3750002651633458, ((1,), 3): 0.3750002651633458, ((2,), 1): 0.3333333333333333, ((2,), 2): 0.3333333333333333, ((2,), 3): 0.3333333333333333}
    print(bfgs_algorithm_for_maximum_entropy_model(dataset[0], dataset[1], [f2]))
