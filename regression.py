from function import IdealFunction

def loss_minize(func_train, func_cand_list, func_los):
    
    func_small_error = None
    err_small = None
    for function in func_cand_list:
        error = func_los(func_train, function)
        if ((err_small == None) or error < err_small):
            err_small = error
            func_small_error = function

    ideal_function = IdealFunction(function=func_small_error, func_train=func_train,
                          error=err_small)
    return ideal_function


def class_find(point, ideal_functions):
   
    class_low_curr = None
    dist_low_curr = None

    for ideal_function in ideal_functions:
        try:
            locate_y_in_classification = ideal_function.locate_y_based_on_x(point["x"])
        except IndexError:
            print("This classification function does not have this point")
            raise IndexError

        distance = abs(locate_y_in_classification - point["y"])

        if (abs(distance < ideal_function.tolerance)):
            if ((class_low_curr == None) or (distance < dist_low_curr)):
                class_low_curr = ideal_function
                dist_low_curr = distance

    return class_low_curr, dist_low_curr
