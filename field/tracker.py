import numpy as np
import warnings
import pandas as pd
from mylib.local_path import f1, f_CellReg_day, f3, f4
from mylib.statistic_test import GetMultidayIndexmap, ReadCellReg
from mylib.multiday.core import MultiDayCore
from mylib.maze_utils3 import field_reallocate, GetDMatrices
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
        self._curr_ref_field = curr_session # A reference field that are used for measuring overlaps.
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
    
    """    
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
    """
    
    def identify_putative_samefield(self, place_field: dict | None) -> tuple[bool, list[int], list[float]]:
        """
        Identify putative same fields that is overlap with the previous one.
        
        Returns
        -------
        bool: Is there any field be identifyed as the same field.
        int or None: return the center of place field which is the most likely to be the same one
        as (highest overlap)
        np.ndarray or None: return field area
        """
        # If the neuron is not detected, the field is natually no detected subsequently.
        if place_field is None:
            return False, np.nan, np.nan
        
        match_fields = []
        match_extent = []
        match_area = []
        # Run for all field candidates to identify if they were the same one as this field.
        for k in place_field.keys():
            is_same, overlap_frac = self._is_samefield(place_field[k])
            if is_same:
                match_fields.append(k)
                match_extent.append(overlap_frac)
                match_area.append(place_field[k])
        
        if len(match_fields) == 0:
            return False, np.nan, np.nan
        else:
            maximum_idx = np.argmax(match_extent)
            return True, match_fields[maximum_idx], match_area[maximum_idx]
    
    def update(
        self, 
        curr_session: int, 
        field_center: int | float | None, 
        field_area: np.ndarray | None,
        is_matched_to_merged_fields: bool = False
    ) -> None:
        self._curr_session += 1
        
        if field_center is None or np.isnan(field_center):
            self._area.append(None)
            self._center.append(np.nan)
        else:
            self._area.append(field_area)
            self._center.append(field_center)
            self._is_active[curr_session] = 1
            
            if is_matched_to_merged_fields == False:
                # Means the newly integrated field is not a merged field.
                self._curr_ref_field = curr_session
            
        
    def _is_overlap(
        self, 
        field_area:np.ndarray, 
        input_area: np.ndarray
    ) -> bool:
        overlap = np.intersect1d(field_area, input_area)
        return len(overlap)/len(field_area) > self._thre or len(overlap)/len(input_area) > self._thre
    
    def _get_overlap(
        self, 
        field_area:np.ndarray, 
        input_area: np.ndarray
    ) -> float:
        overlap = np.intersect1d(field_area, input_area)
        return len(overlap)/len(field_area)
    
    def _is_samefield(self, area: np.ndarray) -> tuple[bool, float]:
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
        float
            The extent of overlap.

        Raises
        ------
        ZeroDivisionError
            If there's no active history of this field, indicating somewhere
            is wrong.
        """
        active_idx = np.flip(np.where(self._is_active == 1)[0])
        
        if active_idx.shape[0] == 0:
            raise ZeroDivisionError(f"There's no active history of this field, indicating somewhere is wrong.")
        
        for i in active_idx:
            if i > self._curr_ref_field:
                continue
            
            if self._is_overlap(self._area[i], area):
                return True, self._get_overlap(self._area[i], area)
        
        return False, self._get_overlap(self._area[self._curr_ref_field], area)
    
        # Criterion 1: Have 60% or more overlapping with the most recently detected fields
        return self._is_overlap(self._area[self._curr_ref_field], area), self._get_overlap(self._area[self._curr_ref_field], area)
        if self._is_overlap(self._area[active_idx[-1]], area) == False:
            return False
        """
        # Criterion 2: It should have overlap with 60% or more the remaining detected fields.
        prev_overlap = 1
        for i in range(len(active_idx)):
            if np.intersect1d(self._area[active_idx[i]], area).shape[0] > 0:
                prev_overlap += 1
        
        if prev_overlap/active_idx.shape[0] < 0.6:
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
        maze_type: int = 1,
        overlap_thre: float = 0.5
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
        self.maze_type = maze_type
        self._generate_fields()
        
    def _generate_fields(self):
        self._indept_fields: list[Field] = []
        self._isdetected = np.zeros(self._total_sesseion, dtype=np.float64)
        
        for i in range(self._total_sesseion):
            if self._place_fields[i] is None:
                # The neuron is not detected in this sessiion
                self._isdetected[i] = np.nan
        
        # Initial Fields
        for i in range(self._total_sesseion):
            if self._place_fields[i] is not None: 
                # Find the first session that this neuron is not inactive.
                if len(self._place_fields[i].keys()) != 0:
                    # The active neuron should have at least 1 field
                    start_session = i
                    # Input all the field(s) this initial cell has for neuron tracking.
                    # This field(s) is/are root field(s)
                    for k in self._place_fields[i].keys():
                        self._indept_fields.append(Field(curr_session=i, 
                                                     area=self._place_fields[i][k], 
                                                     center=k,
                                                     total_sesseion=self._total_sesseion, 
                                                     overlap_thre=self._thre))
                    break
        
        try:
            # Figure out whether this neuron is forever inactive during training.
            # If so, the start_session is not assigned, raising related errors and
            # set the is_silent_neuron to True
            start_session
        except:
            self.is_silent_neuron = True
            return 
        
        # Tracking Fields across the following sessions.
        for i in range(start_session + 1, self._total_sesseion):
            match_fields = [] # Matched fields in session i
            match_area = []
            for pf in self._indept_fields:
                mat, field_center, field_area = pf.identify_putative_samefield(self._place_fields[i])

                match_fields.append(field_center)
                match_area.append(field_area)
            
            # Update previous fields

            for n, pf in enumerate(self._indept_fields):
                if np.isnan(match_fields[n]):
                    pf.update(i, match_fields[n], match_area[n])
                else:
                    pf.update(i, match_fields[n], match_area[n], 
                              match_fields.count(match_fields[n]) > 1)
            
            # unmatched fields in session i
            match_fields = np.array(match_fields)
            match_fields = match_fields[np.where(np.isnan(match_fields) == False)[0]]
            
            if self._place_fields[i] is None:
                continue
            
            nomatch_fields = np.setdiff1d(np.array([k for k in self._place_fields[i].keys()]), match_fields)

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
        field_info = np.zeros((self._total_sesseion, len(self._indept_fields), 5), dtype=np.float64)
        # The five dimensions contain information: cell_id, is place cell, field center, field size, and distance to the entry.
        # If no detected, all will be nan. If no field, only field center, and field size will be nan.
        
        if self.is_silent_neuron:
            return field_reg*np.nan, field_info*np.nan
        
        D = GetDMatrices(maze_type=self.maze_type, nx = 48)
        field_info[:, :, 4] = np.nan
        
        for i, pf in enumerate(self._indept_fields):
            field_reg[:, i] = pf.register(self._isdetected)
            field_info[:, i, 0] = self._id
            field_info[:, i, 1] = self._is_placecell
            field_info[:, i, 2] = pf.get_fieldcenter()
            field_info[:, i, 3] = pf.get_fieldsize()
            field_info[np.where(np.isnan(field_info[:, i, 2]) == False)[0], i, 4] = D[0, field_info[np.where(np.isnan(field_info[:, i, 2]) == False)[0], i, 2].astype(int)-1]

        
        mean_distance = np.nanmean(field_info[:, :, 4], axis=0)
        median_distance = np.nanmedian(field_info[:, :, 4], axis=0)
        sort_indices = np.argsort(mean_distance)
        field_reg = field_reg[:, sort_indices]
        field_info = field_info[:, sort_indices, :]
        return field_reg, field_info

    def _merge_columns(self, matrix):
        return np.apply_along_axis(lambda row: row[~np.isnan(row)][0] if np.any(~np.isnan(row)) else np.nan, 1, matrix)

    def _merge_fields(
        self,
        field_reg: np.ndarray,
        field_info: np.ndarray,
        smooth_map_all: np.ndarray
    ):
        """
        Two fields that have been detected to merge for once were merged together.
        
        The merge matrix denote the relationship between each identified raw field
        >>> # It is a block matrix:
        >>> merge_matrix = np.array([[1, 1, 1, 0, 0, 0],
                                     [1, 1, 1, 0, 0, 0],
                                     [1, 1, 1, 0, 0, 0],
                                     [0, 0, 0, 1, 1, 0],
                                     [0, 0, 0, 1, 1, 0],
                                     [0, 0, 0, 0, 0, 1]])
        """
        merge_matrix = np.zeros((field_reg.shape[1], field_reg.shape[1]))
        for i in range(field_reg.shape[1]-1):
            for j in range(i+1, field_reg.shape[1]):
                if np.where(field_info[:, i, 2] - field_info[:, j, 2] == 0)[0].shape[0] >= 1: # Merged for two times field
                    merge_matrix[i, j] = merge_matrix[j, i] = 1
                    merge_matrix[i, i] = merge_matrix[j, j] = 1
                    merge_matrix[j, :] += merge_matrix[i, :]
                    merge_matrix[i, :] = merge_matrix[j, :]
                    merge_matrix[:, j] += merge_matrix[:, i]   
                    merge_matrix[:, i] = merge_matrix[:, j]
                         
        merge_matrix[merge_matrix > 1] = 1
        continue_border = merge_matrix.shape[0]
        for i in range(merge_matrix.shape[0]-1, -1, -1):
            if i >= continue_border:
                continue
            
            transition = np.ediff1d(np.concatenate([[0], merge_matrix[i, :], [0]]))
            if np.where(transition != 0)[0].shape[0] == 0:
                continue # No need to merge
            else:
                start_idx = np.where(transition == 1)[0][0]
                end_idx = np.where(transition == -1)[0][-1]
                continue_border = start_idx

                # Merge fields between start_idx and end_idx
                merged_reg = np.sum(field_reg[:, start_idx:end_idx], axis=1)
                merged_reg[merged_reg > 1] = 1
                field_reg = np.delete(field_reg, np.arange(start_idx+1, end_idx), axis=1)
                                       
                field_reg[:, start_idx] = merged_reg
                
                # Update place field all and update field info
                field_centers = field_info[:, start_idx:end_idx, 2]
                for j in range(field_centers.shape[0]):
                    idx = np.where(np.isnan(field_centers[j, :]) == False)[0]
                    present_centers = np.unique(field_centers[j, np.where(np.isnan(field_centers[j, :]) == False)[0]].astype(int))
                    if len(present_centers) == 1:
                        field_info[j, start_idx, 2] = present_centers[0]
                        field_info[j, start_idx, 3] = field_info[j, start_idx + idx[0], 3]
                        field_info[j, start_idx, 4] = field_info[j, start_idx + idx[0], 4]
                        
                    elif len(present_centers) > 1:
                        field_area = np.concatenate([self._place_fields[j][k] for k in present_centers])
                        for k in present_centers:
                            del self._place_fields[j][k]
                        
                        arg_max = np.argmax(smooth_map_all[j, present_centers-1])
                        
                        self._place_fields[j][present_centers[arg_max]] = field_area
                        field_info[j, start_idx, 2] = present_centers[arg_max]
                        field_info[j, start_idx, 3] = len(field_area)
                        field_info[j, start_idx, 4] = field_info[j, start_idx + idx[arg_max], 4]
                
                field_info = np.delete(field_info, np.arange(start_idx+1, end_idx), axis=1)

        return field_reg, field_info, self._place_fields

    @staticmethod
    def field_register(
        index_map: np.ndarray,
        place_field_all: list[list[dict]],
        is_placecell: np.ndarray,
        smooth_map_all: np.ndarray,
        overlap_thre: float = 0.6,
        maze_type: int = 1
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
        field_info = np.zeros((is_placecell.shape[0], 1, 5), np.float64)
        
        index_map = index_map.astype(np.int64)
        
        for i in tqdm(range(is_placecell.shape[1])):
            neuron = Tracker(
                neuron_id=index_map[:, i],
                place_fields=place_field_all[i],
                is_placecell=is_placecell[:, i],
                total_sesseion=index_map.shape[0],
                overlap_thre=overlap_thre,
                maze_type=maze_type
            )
            reg, info = neuron.register()
            reg, info, place_field_all[i] = neuron._merge_fields(
                reg, info,
                smooth_map_all=smooth_map_all[:, i, :])
            field_reg = np.concatenate([field_reg, reg], axis=1)
            field_info = np.concatenate([field_info, info], axis=1)
        
        field_reg, field_info = field_reg[:, 1:], field_info[:, 1:, :]
        num = np.nansum(field_reg, axis=0)
        idx = np.where(np.isnan(num) == False)[0]
        
        print(field_reg.shape, field_info.shape)

        return field_reg[:, idx], field_info[:, idx, :], place_field_all

def get_field_ids(field_info: np.ndarray) -> np.ndarray:
    column_dict = {}
    labels = []
    for col in range(field_info.shape[1]):
        column = tuple(field_info[:, col, 0])  # Convert column to a tuple to make it hashable
        if column not in column_dict:
            column_dict[column] = len(column_dict) + 1  # Assign a new ID
        labels.append(column_dict[column])
    return np.array(labels)

def get_field_centers(field_info: np.ndarray, maze_type: int) -> np.ndarray:
    D = GetDMatrices(maze_type, 48)
    
    field_centers = np.zeros(field_info.shape[1])
    for col in range(field_info.shape[1]):
        distances = D[0, field_info[:, col, 2][np.where(np.isnan(field_info[:, col, 2]) == False)[0]].astype(int)-1]
        field_centers[col] = field_info[:, col, 2][np.where(np.isnan(field_info[:, col, 2]) == False)[0]][np.where(distances >= np.median(distances))[0][0]]
    
    return field_centers

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
    behavior_paradigm: str | None = None,
    is_shuffle: bool = False,
    prefix: str = 'trace_mdays_conc'
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
    
    index_map = index_map[1:, :]
        
    # Initial basic elements
    n_neurons = index_map.shape[1]
    n_sessions = index_map.shape[0]    

    # Get information from daily trace.pkl
    core = MultiDayCore(
        keys = ['is_placecell', 'place_field_all_multiday', 'smooth_map_all'],
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
    
    file_indices = file_indices[1:]  
    print(file_indices, mouse, stage, session)
    res = core.get_trace_set(f=fdata, file_indices=file_indices, keys=['is_placecell', 'place_field_all_multiday', 'smooth_map_all'])
    
    is_placecell = np.full((n_sessions, n_neurons), np.nan)
    smooth_map_all = np.full((n_sessions, n_neurons, 2304), np.nan)
    place_field_all = [[] for _ in range(n_neurons)]
    for j in range(n_neurons):
        for i in range(n_sessions):
            if index_map[i, j] > 0:
                is_placecell[i, j] = res['is_placecell'][i][int(index_map[i, j])-1]
                smooth_map_all[i, j, :] = res['smooth_map_all'][i][int(index_map[i, j])-1, :]
                
                if is_shuffle == False:
                    place_field_all[j].append(res['place_field_all_multiday'][i][int(index_map[i, j])-1])
                else:
                    place_field_all[j].append(field_reallocate(res['place_field_all_multiday'][i][int(index_map[i, j])-1], maze_type=maze_type))
            else:
                place_field_all[j].append(None)
    print("Field Register...")            
    field_reg, field_info, place_field_all = Tracker.field_register(
        index_map=index_map,
        place_field_all=place_field_all,
        is_placecell=is_placecell,
        overlap_thre=overlap_thre,
        maze_type=maze_type,
        smooth_map_all=smooth_map_all
    )
    field_ids = get_field_ids(field_info)
    shuffle_type = 'Shuffle' if is_shuffle else 'Real'
    
    trace = {"MiceID": mouse, "Stage": stage, "session": session, "maze_type": maze_type, "paradigm": behavior_paradigm,
             "is_placecell": is_placecell, "place_field_all": place_field_all, "field_reg": field_reg, "field_info": field_info, 
             "field_ids": field_ids, 'is_shuffle': shuffle_type, "field_centers": get_field_centers(field_info, maze_type),
              "n_neurons": n_neurons, "n_sessions": n_sessions, "maze_type": maze_type,
             "index_map": index_map.astype(np.int64)}

    appendix = '' if is_shuffle == False else '_shuffle'
    with open(os.path.join(os.path.dirname(cellreg_dir), prefix+appendix+".pkl"), 'wb') as handle:
        print(os.path.join(os.path.dirname(cellreg_dir), prefix+appendix+".pkl"))
        pickle.dump(trace, handle)
        
    del res
    gc.collect()
    
    if behavior_paradigm == 'CrossMaze':
        del trace
        return
    else:
        DATA = {"MiceID": mouse, "Stage": stage, "session": session, "maze_type": maze_type, "paradigm": behavior_paradigm,
                "cis":{"is_placecell": is_placecell, "place_field_all": place_field_all, "field_reg": field_reg, "field_info": field_info, 'field_ids': field_ids,
                       "field_centers": get_field_centers(field_info, maze_type)},
                "n_neurons": n_neurons, "n_sessions": n_sessions, "maze_type": maze_type, 'is_shuffle': shuffle_type,
                "index_map": index_map.astype(np.int64)}
    
    n_neurons = index_map.shape[1]
    n_sessions = index_map.shape[0]    

    # Get information from daily trace.pkl
    core = MultiDayCore(
        keys = ['is_placecell', 'place_field_all_multiday', 'smooth_map_all'],
        paradigm=behavior_paradigm,
        direction='trs'
    )
        
    res = core.get_trace_set(f=fdata, file_indices=file_indices, keys=['is_placecell', 'place_field_all_multiday', 'smooth_map_all'])
    
    is_placecell = np.full((n_sessions, n_neurons), np.nan)
    place_field_all = [[] for _ in range(n_neurons)]
    smooth_map_all = np.full((n_sessions, n_neurons, 2304), np.nan)
    
    for j in range(n_neurons):
        for i in range(n_sessions):
            if index_map[i, j] > 0:
                is_placecell[i, j] = res['is_placecell'][i][int(index_map[i, j])-1]
                smooth_map_all[i, j, :] = res['smooth_map_all'][i][int(index_map[i, j])-1, :]
                if is_shuffle == False:
                    place_field_all[j].append(res['place_field_all_multiday'][i][int(index_map[i, j])-1])
                else:
                    place_field_all[j].append(field_reallocate(res['place_field_all_multiday'][i][int(index_map[i, j])-1], maze_type=maze_type))
            else:
                place_field_all[j].append(None)
    print("Field Register...")            
    field_reg, field_info, place_field_all = Tracker.field_register(
        index_map=index_map,
        place_field_all=place_field_all,
        is_placecell=is_placecell,
        overlap_thre=overlap_thre,
        maze_type=maze_type,
        smooth_map_all=smooth_map_all
    )
    field_ids = get_field_ids(field_info)
    
    DATA['trs'] = {"is_placecell": is_placecell, "place_field_all": place_field_all, "field_reg": field_reg, "field_info": field_info, 'field_ids': field_ids,
                   "field_centers": get_field_centers(field_info, maze_type)}
    with open(os.path.join(os.path.dirname(cellreg_dir), prefix+appendix+".pkl"), 'wb') as handle:
        print(os.path.join(os.path.dirname(cellreg_dir), prefix+appendix+".pkl"))
        pickle.dump(DATA, handle)


if __name__ == "__main__":  
    import pickle
    from mylib.local_path import f_CellReg_modi as f
    from tqdm import tqdm

    for i in tqdm(range(len(f))):
        if f['include'][i] == 0:
            continue
        
        if i != 23:
            continue
        
        """
        with open(f['Trace File'][i], 'rb') as handle:
            trace = pickle.load(handle)
        
        j = 50
        idx = np.where(trace['field_ids'] == j)[0]
        print(trace['field_ids'][idx])
        print(trace['field_info'][:, idx, 0])

        """
        is_shuffle = f['Type'][i] == 'Shuffle'
        

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
        """

        # CellReg
        try:
            index_map = GetMultidayIndexmap(
                    mouse=f['MiceID'][i],
                    stage=f['Stage'][i],
                    session=f['session'][i],
                    occu_num=2
            )            
        except:
            index_map = ReadCellReg(f['cellreg_folder'][i])
        """        
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
            behavior_paradigm=f['paradigm'][i],
            is_shuffle=is_shuffle,
            prefix='trace_mdays_conc',
        )
