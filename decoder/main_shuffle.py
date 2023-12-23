from mylib.decoder.main import main, f1
import pandas as pd

res = pd.read_excel(r"E:\Data\Simulation_pc\cross_maze_decode.xlsx", sheet_name='Shuffle')
j = 3734
for i in range(382, len(f1)):
    res, j = main(f1, i, res, j, save_loc = r"E:\Data\Simulation_pc\cross_maze_decode_shuffle", 
                  save_sheet=r"E:\Data\Simulation_pc\cross_maze_decode_shuffle_output.xlsx", 
                  is_shuffle=True)