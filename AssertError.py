# Detect if key is already existed in certain directory.
def KeyWordErrorCheck(trace:dict, file_loc:str, keys:list = []):
    is_correct = True
    for k in keys:
        try: trace[k]
        except: 
            print("KeyWordError! Input dict lack a crucial key '"+str(k)+"'. Reported by "+file_loc)
            is_correct = False
            
    assert is_correct

# Interface specific error:
def VariablesInputErrorCheck(input_variable:list = [], check_variable:list = []):
    if input_variable == check_variable:
        return True
    else:
        print("VariablesInputError! You need to check the input variable_name! or it'll return a ERROR!")
        print("The correct form is:",check_variable)
        assert False

# Report location of error.
def ReportErrorLoc(loc:str):
    '''
    Note: Report an error, print the input string and assert the code.
    '''
    print(loc)
    assert False

def ValueErrorCheck(input_variable, check_variable: list = []):
    if input_variable in check_variable:
        return True
    else:
        print("ValueError! the input variable is not valid! Only the following values are valid:\n",check_variable)
        assert False