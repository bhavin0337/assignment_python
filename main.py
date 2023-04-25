import math
from function import FunctionManager
from regression import loss_minize, class_find
from lossfunction import squared_error
from plotting import func_plot_ideal, func_ideal_with_points_plot
from utils import results_to_sqlite

ACCEPTED_FACTOR = math.sqrt(2)

if __name__ == '__main__':
    ideal_path = "data/ideal.csv"
    train_path = "data/train.csv"

    man_ideal_func_candi = FunctionManager(path_of_csv=ideal_path)
    func_man_train = FunctionManager(path_of_csv=train_path)

    func_man_train.to_sql(file_name="training", suffix=" (training func)")
    man_ideal_func_candi.to_sql(file_name="ideal", suffix=" (ideal func)")

    ideal_functions = []
    for train_function in func_man_train:
        ideal_function = loss_minize(func_train=train_function,
                                       func_cand_list=man_ideal_func_candi.functions,
                                       func_los=squared_error)
        ideal_function.tolerance_factor = ACCEPTED_FACTOR
        ideal_functions.append(ideal_function)

    func_plot_ideal(ideal_functions, "train_and_ideal")

    test_path = "data/test.csv"
    test_function_manager = FunctionManager(path_of_csv=test_path)
    test_function = test_function_manager.functions[0]

    points_with_ideal_function = []
    for point in test_function:
        ideal_function, delta_y = class_find(point=point, ideal_functions=ideal_functions)
        result = {"point": point, "classification": ideal_function, "delta_y": delta_y}
        points_with_ideal_function.append(result)

    func_ideal_with_points_plot(points_with_ideal_function, "point_and_ideal")

    results_to_sqlite(points_with_ideal_function)
    

