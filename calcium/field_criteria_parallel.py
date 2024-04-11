import multiprocessing
from tqdm import tqdm
from mylib.calcium.field_criteria import GetPlaceField

def _process_place_field(args):
    trace, k, thre_type, parameter, events_num_crit, need_events_num, split_thre = args
    return GetPlaceField(
        trace=trace, 
        n=k, 
        thre_type=thre_type, 
        parameter=parameter, 
        events_num_crit=events_num_crit, 
        need_events_num=need_events_num,
        split_thre=split_thre
    )

def place_field_parallel(trace: dict, thre_type: int = 2, parameter: float = 0.4, events_num_crit: int = 10, need_events_num: bool = True, split_thre: float = 0.2, n_jobs: int = -1):
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    pool = multiprocessing.Pool(processes=n_jobs)
    args_list = [(trace, k, thre_type, parameter, events_num_crit, need_events_num, split_thre) for k in range(trace['n_neuron'])]
    
    place_field_all = []
    for result in tqdm(pool.imap_unordered(_process_place_field, args_list), total=len(args_list)):
        place_field_all.append(result)
    pool.close()
    pool.join()

    print("    Place field has been generated successfully.")
    return place_field_all
