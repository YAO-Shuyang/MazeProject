import numpy as np
import warnings
import pandas as pd
from mylib.local_path import f1, f_CellReg_day, f3, f4
from mylib.statistic_test import GetMultidayIndexmap, ReadCellReg
from mylib.multiday.core import MultiDayCore
from mylib.local_path import f1, f_CellReg_day
from tqdm import tqdm
import os
import pickle
import gc

class Field(object):
    def __init__(
        self, 
        curr_session: int, 
        area: np.ndarray, 
        center: np.ndarray,
        total_sesseion: int = 26,
        overlap_thre: float = 0.6
    ) -> None:
        """__init__

        Parameters
        ----------
        curr_session : int
            The first session that a field was detected
        area : np.ndarray
            The covering area of this field
        total_sesseion : int, optional
            The total number of sessions, by default 26
        """
        self._is_active = np.zeros(total_sesseion, dtype=np.float64)
        self._is_active[curr_session] = 1
        self._area = [None for i in range(curr_session)] + [area]
        self._center = [None for i in range(curr_session)] + [center]
        self._curr_session = curr_session
        self._total_session = total_sesseion
        self._thre = overlap_thre
        
    @property
    def curr_session(self):
        return self._curr_session
    
    @property
    def area(self):
        return self._area
    
    @property
    def center(self):
        return self._center
    
    @property
    def is_active(self):
        return self._is_active
        
    def update(self, curr_session: int, place_field: dict | None) -> bool:
        self._curr_session += 1
        
        # If the neuron is not detected, the field is natually no detected subsequently.
        if place_field is None:
            self._area.append(None)
            self._center.append(np.nan)
            return False, []
        
        match_fields = []
        # Run for all field candidates to identify if they were the same one as this field.
        for k in place_field.keys():
            if self._is_samefield(place_field[k]):
                self._is_active[curr_session] = 1
                self._area.append(place_field[k])
                self._center.append(k)
                match_fields.append(k)
                return True, match_fields
            
        self._area.append(None)
        self._center.append(np.nan)  
        return False, []
        
    def _is_overlap(
        self, 
        field_area:np.ndarray, 
        input_area: np.ndarray
    ) -> bool:
        overlap = np.intersect1d(field_area, input_area)
        return len(overlap)/len(field_area) >= self._thre or len(overlap)/len(input_area) >= self._thre
    
    def _is_samefield(self, area: np.ndarray) -> bool:
        """_is_samefield: function to identify if the inputed field is the 
        same one of this field.

        Parameters
        ----------
        area : np.ndarray
            The area of the inputed field

        Returns
        -------
        bool
            Whether the inputed field is the same one of this field

        Raises
        ------
        ZeroDivisionError
            If there's no active history of this field, indicating somewhere
            is wrong.
        """
        active_idx = np.where(self._is_active == 1)[0]
        
        if active_idx.shape[0] == 0:
            raise ZeroDivisionError(f"There's no active history of this field, indicating somewhere is wrong.")
        
        # Criterion 1: Have 60% or more overlapping with the most recently detected fields
        return self._is_overlap(self._area[active_idx[-1]], area)
        
        """
        # Criterion 2: It should have overlap with 50% or more the remaining detected fields.
        prev_overlap = 1
        for i in range(len(active_idx)):
            if np.intersect1d(self._area[active_idx[i]], area).shape[0] > 0:
                prev_overlap += 1
        
        if prev_overlap/active_idx.shape[0] < 0.2:
            return False
        else:  
            return True
        """
    def register(self, is_detected):
        if self.curr_session != self._total_session-1:
            warnings.warn(
                f"Current session is {self.curr_session+1}, but total sessions is {self._total_session}."
            )
        return self.is_active + is_detected

    def get_fieldcenter(self):
        return np.array(self._center, dtype=np.float64)
    
    def get_fieldsize(self):
        size = np.zeros(self._total_session, dtype=np.float64)
        for i in range(self._total_session):
            if self._area[i] is None:
                size[i] = np.nan
            else:
                size[i] = len(self._area)
        return size
        

class Tracker(object):
    def __init__(
        self, 
        neuron_id: np.ndarray,
        place_fields: list[dict], 
        is_placecell: np.ndarray,
        total_sesseion: int = 26,
        overlap_thre: float = 0.6
    ) -> None:
        """__init__: Initialize Tracker

        Parameters
        ----------
        neuron_id: np.ndarray,
            the id of the registered neurons in each session. 
        place_fields : list[dict]
            A list contain the detected place fields in all the n sessions.
        smooth_maps : np.ndarray
            The spatial rate map of the input neuron over all sessions
        is_placecell : np.ndarray, shape (n_sessions, 2304), if the neuron is not detected
            in any session, the then the vector smooth_maps[i, :] is filled with np.nan.
            Whether this neuron is a place cell in each session or not.
        total_sesseion : int, optional
            The total number of sessions, by default 26
        """
        assert is_placecell.shape[0] == total_sesseion
        assert len(place_fields) == total_sesseion
        
        self.is_silent_neuron = False
        self._id = neuron_id
        self._total_sesseion = total_sesseion
        self._place_fields = place_fields
        self._is_placecell = is_placecell
        self._thre = overlap_thre
        self._generate_fields()
        
    def _generate_fields(self):
        self._indept_fields: list[Field] = []
        self._isdetected = np.zeros(self._total_sesseion, dtype=np.float64)
        
        for i in range(self._total_sesseion):
            if self._place_fields[i] is None:
                # The neuron is not detected in this sessiion
                self._isdetected[i] = np.nan
        
        for i in range(self._total_sesseion):
            if self._place_fields[i] is not None:
                if len(self._place_fields[i].keys()) != 0:
                    start_session = i
                    for k in self._place_fields[i].keys():
                        self._indept_fields.append(Field(curr_session=i, 
                                                     area=self._place_fields[i][k], 
                                                     center=k,
                                                     total_sesseion=self._total_sesseion, 
                                                     overlap_thre=self._thre))
                    break
        
        try:
            start_session
        except:
            self.is_silent_neuron = True
            return 
        
        #update
        
        for i in range(start_session + 1, self._total_sesseion):
            match_fields = []
            for pf in self._indept_fields:
                mat, fields = pf.update(i, self._place_fields[i])
                if mat:
                    match_fields = match_fields + fields
                    
            if self._place_fields[i] is None:
                continue
            
            nomatch_fields = np.setdiff1d(np.array([k for k in self._place_fields[i].keys()]), np.array(match_fields))

            for k in nomatch_fields:
                self._indept_fields.append(Field(
                    curr_session=i, 
                    area=self._place_fields[i][k],
                    center=k,
                    total_sesseion=self._total_sesseion,
                    overlap_thre=self._thre
                ))
                
    def register(self) -> tuple[np.ndarray, np.ndarray]:
        """register: Register place fields

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Return two matrix, representing the field_reg and field_info, respectively.
        """
        field_reg = np.zeros((self._total_sesseion, len(self._indept_fields)), dtype=np.float64)
        field_info = np.zeros((self._total_sesseion, len(self._indept_fields), 4), dtype=np.float64)
        # The five dimensions contain information: cell_id, is place cell, field center, field size,.
        # If no detected, all will be nan. If no field, only field center, and field size will be nan.
        
        if self.is_silent_neuron:
            return field_reg*np.nan, field_info*np.nan

        
        for i, pf in enumerate(self._indept_fields):
            field_reg[:, i] = pf.register(self._isdetected)
            field_info[:, i, 0] = self._id
            field_info[:, i, 1] = self._is_placecell
            field_info[:, i, 2] = pf.get_fieldcenter()
            field_info[:, i, 3] = pf.get_fieldsize()
            
        return field_reg, field_info
    
    @staticmethod
    def field_register(
        index_map: np.ndarray,
        place_field_all: list[list[dict]],
        is_placecell: np.ndarray,
        overlap_thre: float = 0.6
    ) -> tuple[np.ndarray, np.ndarray]:
        """field_register: Register all place fields from the registered neuron

        Parameters
        ----------
        index_map : np.ndarray, shape (n_session, n_neuron)
            The indices of registered neurons.
        place_field_all : list[list[dict]]
            A list contains n_neuron lists, and each of the inside list has a 
            length of n_sessions that contains the place fields of this neuron
            in each recording session.
        is_placecell : np.ndarray, shape (n_session, n_neuron)
            The identity of the detected neuron on each day.
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            return the whole field_reg and field_info
        """
        assert len(place_field_all) == is_placecell.shape[1]
        
        field_reg = np.zeros((is_placecell.shape[0], 1), np.float64)
        field_info = np.zeros((is_placecell.shape[0], 1, 4), np.float64)
        
        index_map = index_map.astype(np.int64)
        
        for i in tqdm(range(is_placecell.shape[1])):
            neuron = Tracker(
                neuron_id=index_map[:, i],
                place_fields=place_field_all[i],
                is_placecell=is_placecell[:, i],
                total_sesseion=index_map.shape[0],
                overlap_thre=overlap_thre
            )
            reg, info = neuron.register()
            field_reg = np.concatenate([field_reg, reg], axis=1)
            field_info = np.concatenate([field_info, info], axis=1)
        
        field_reg, field_info = field_reg[:, 1:], field_info[:, 1:, :]
        num = np.nansum(field_reg, axis=0)
        idx = np.where(np.isnan(num) == False)[0]
        
        print(field_reg.shape, field_info.shape)

        return field_reg[:, idx], field_info[:, idx, :]
    

def main(
    i: int,
    f: pd.DataFrame = f_CellReg_day,
    overlap_thre: float = 0.6,
    index_map: np.ndarray | None = None,
    cellreg_dir: str | None = None,
    mouse: int | None = None,
    stage: str | None = None,
    session: int | None = None,
    maze_type: int | None = None,
    behavior_paradigm: str | None = None
):

    
    if index_map is None:
        if f['maze_type'][i] == 0:
            return f
        line = i
        cellreg_dir = f['cellreg_folder'][i]
        mouse = int(f['MiceID'][i])
        stage = f['Stage'][i]
        session = int(f['session'][i])
        maze_type = int(f['maze_type'][i])
        behavior_paradigm = f['paradigm'][i]
    
        index_map = GetMultidayIndexmap(
            mouse,
            stage=stage,
            session=session,
            i = i,
            occu_num=2
        )
        index_map[np.where((np.isnan(index_map))|(index_map < 0))] = 0
    else:
        index_map[np.where((np.isnan(index_map))|(index_map < 0))] = 0
        assert mouse is not None
        assert stage is not None
        assert session is not None
        assert maze_type is not None
        assert behavior_paradigm is not None
        assert cellreg_dir is not None
        direction = None if behavior_paradigm == 'CrossMaze' else 'cis'
        
    if behavior_paradigm == 'CrossMaze':
        fdata = f1
    elif behavior_paradigm == 'ReverseMaze':
        fdata = f3
    elif behavior_paradigm == 'HairpinMaze':
        fdata = f4
    else:
        raise ValueError(f"Paradigm {behavior_paradigm} is not supported.")
        
        
    # Initial basic elements
    n_neurons = index_map.shape[1]
    n_sessions = index_map.shape[0]    

    # Get information from daily trace.pkl
    core = MultiDayCore(
        keys = ['is_placecell', 'place_field_all_multiday'],
        paradigm=behavior_paradigm,
        direction=direction
    )
    file_indices = np.where((fdata['MiceID'] == mouse) & (fdata['Stage'] == stage) & (fdata['session'] == session))[0]
    
    if mouse in [11095, 11092]:
        file_indices = file_indices[3:]
    
    if stage == 'Stage 1+2':
        file_indices = np.where((fdata['MiceID'] == mouse) & (fdata['session'] == session) & ((fdata['Stage'] == 'Stage 1') | (fdata['Stage'] == 'Stage 2')))[0]
        
    if stage == 'Stage 1' and mouse in [10212] and session == 2:
        file_indices = np.where((fdata['MiceID'] == mouse) & (fdata['session'] == session) & (fdata['Stage'] == 'Stage 1') & (fdata['date'] != 20230506))[0]
        
    print(file_indices, mouse, stage, session)
    res = core.get_trace_set(f=fdata, file_indices=file_indices, keys=['is_placecell', 'place_field_all_multiday'])
    
    is_placecell = np.full((n_sessions, n_neurons), np.nan)
    place_field_all = [[] for _ in range(n_neurons)]
    for j in range(n_neurons):
        for i in range(n_sessions):
            if index_map[i, j] > 0:
                is_placecell[i, j] = res['is_placecell'][i][int(index_map[i, j])-1]
                place_field_all[j].append(res['place_field_all_multiday'][i][int(index_map[i, j])-1])
            else:
                place_field_all[j].append(None)
    print("Field Register...")            
    field_reg, field_info = Tracker.field_register(
        index_map=index_map,
        place_field_all=place_field_all,
        is_placecell=is_placecell,
        overlap_thre=overlap_thre
    )
    
    trace = {"MiceID": mouse, "Stage": stage, "session": session, "maze_type": maze_type, "paradigm": behavior_paradigm,
             "is_placecell": is_placecell, "place_field_all": place_field_all, "field_reg": field_reg, "field_info": field_info,
              "n_neurons": n_neurons, "n_sessions": n_sessions, "maze_type": maze_type,
             "index_map": index_map.astype(np.int64)}

    with open(os.path.join(os.path.dirname(cellreg_dir), "trace_mdays_conc.pkl"), 'wb') as handle:
        print(os.path.join(os.path.dirname(cellreg_dir), "trace_mdays_conc.pkl"))
        pickle.dump(trace, handle)
        
    del res
    gc.collect()
    
    if behavior_paradigm == 'CrossMaze':
        del trace
        return
    else:
        DATA = {"MiceID": mouse, "Stage": stage, "session": session, "maze_type": maze_type, "paradigm": behavior_paradigm,
                "cis":{"is_placecell": is_placecell, "place_field_all": place_field_all, "field_reg": field_reg, "field_info": field_info},
                "n_neurons": n_neurons, "n_sessions": n_sessions, "maze_type": maze_type,
                "index_map": index_map.astype(np.int64)}
    
    n_neurons = index_map.shape[1]
    n_sessions = index_map.shape[0]    

    # Get information from daily trace.pkl
    core = MultiDayCore(
        keys = ['is_placecell', 'place_field_all_multiday'],
        paradigm=behavior_paradigm,
        direction='trs'
    )
        
    res = core.get_trace_set(f=fdata, file_indices=file_indices, keys=['is_placecell', 'place_field_all_multiday'])
    
    is_placecell = np.full((n_sessions, n_neurons), np.nan)
    place_field_all = [[] for _ in range(n_neurons)]
    for j in range(n_neurons):
        for i in range(n_sessions):
            if index_map[i, j] > 0:
                is_placecell[i, j] = res['is_placecell'][i][int(index_map[i, j])-1]
                place_field_all[j].append(res['place_field_all_multiday'][i][int(index_map[i, j])-1])
            else:
                place_field_all[j].append(None)
    print("Field Register...")            
    field_reg, field_info = Tracker.field_register(
        index_map=index_map,
        place_field_all=place_field_all,
        is_placecell=is_placecell,
        overlap_thre=overlap_thre
    )
    
    DATA['trs'] = {"is_placecell": is_placecell, "place_field_all": place_field_all, "field_reg": field_reg, "field_info": field_info}
    with open(os.path.join(os.path.dirname(cellreg_dir), "trace_mdays_conc.pkl"), 'wb') as handle:
        print(os.path.join(os.path.dirname(cellreg_dir), "trace_mdays_conc.pkl"))
        pickle.dump(DATA, handle)


if __name__ == "__main__":  
    from mylib.local_path import f_CellReg_modi as f
    
    for i in range(len(f)):
        
        if f['paradigm'][i] == 'CrossMaze':
            if f['maze_type'][i] == 0:
                index_map = GetMultidayIndexmap(
                    mouse=f['MiceID'][i],
                    stage=f['Stage'][i],
                    session=f['session'][i],
                    occu_num=2
                )
            else:
                with open(f['cellreg_folder'][i], 'rb') as handle:
                    index_map = pickle.load(handle)
        else:
            index_map = ReadCellReg(f['cellreg_folder'][i])
            
    # with open(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\10224\neuromatch_res.pkl", 'rb') as handle:
    #     index_map = pickle.load(handle)
    
        index_map[np.where((index_map < 0)|np.isnan(index_map))] = 0
        mat = np.where(index_map>0, 1, 0)
        num = np.sum(mat, axis = 0)
        index_map = index_map[:, np.where(num >= 2)[0]]  
        print(index_map.shape)
        main(
            i=i,
            f=f,
            index_map=index_map,
            overlap_thre=0.6,
            cellreg_dir=f['cellreg_folder'][i],
            mouse=f['MiceID'][i],
            stage=f['Stage'][i],
            session=f['session'][i],
            maze_type=f['maze_type'][i],
            behavior_paradigm=f['paradigm'][i]
        )
    """
    with open(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\trace_mdays_conc.pkl", 'rb') as handle:
        trace = pickle.load(handle)
        
    field_reg, field_info = Tracker.field_register(
        index_map=index_map,
        place_field_all=trace['place_field_all'],
        is_placecell=trace['is_placecell']
    )
    print(field_reg[:, :4])
    """