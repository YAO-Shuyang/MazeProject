import numpy as np



# def split_sibling_field_pairs()

if __name__ == '__main__':
    import pickle
    from mylib.field.within_field import within_field_half_half_correlation, within_field_odd_even_correlation
    from mylib.preprocessing_ms import field_register
    from mylib.local_path import f1
    import copy as cp
    
    for i in range(len(f1)):
        print(i, f1['MiceID'][i], f1['date'][i], ' session '+str(f1['session'][i]))
        if f1['include'][i] == 0:
            continue
        
        with open(f1['Trace File'][i], 'rb') as handle:
            trace = pickle.load(handle)
        
        trace['FSCList'] = within_field_half_half_correlation(
            trace['smooth_map_fir'],
            trace['smooth_map_sec'],
            trace['place_field_all']
        )
        trace['OECList'] = within_field_odd_even_correlation(
            trace['smooth_map_odd'],
            trace['smooth_map_evn'],
            trace['place_field_all']
        )
        trace = field_register(trace)
        
        with open(f1['Trace File'][i], 'wb') as handle:
            pickle.dump(trace, handle)
        print()
    
    with open(r"E:\Data\Cross_maze\10227\20230930\session 2\trace.pkl", "rb") as handle:
        trace = pickle.load(handle)
        
    print(trace['field_reg'])