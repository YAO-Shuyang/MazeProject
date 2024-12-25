import numpy as np
import warnings
import pandas as pd
from mylib.local_path import f1, f_CellReg_day, f3, f4
from mylib.statistic_test import GetMultidayIndexmap, ReadCellReg, calc_ratemap
from mylib.multiday.core import MultiDayCore
from mylib.maze_utils3 import field_reallocate, GetDMatrices
from tqdm import tqdm
import os
import pickle
import gc
import matplotlib.pyplot as plt
import time
from mylib.maze_utils3 import Clear_Axes
from mazepy.datastruc.neuact import SpikeTrain
from mazepy.basic._utils import _convert_to_kilosort_form
from mazepy.basic._calc_rate import _get_kilosort_spike_counts

OVERLAP_THRE = 0.75
REITERATE_TIMES = 3
MERGED_FIELD_SUPREME = 10
MERGED_IDENTIFY_THRE = 1
MERGED_OVERLAP_THRE = 0.5

class Field(object):
    def __init__(
        self, 
        curr_session: int, 
        area: np.ndarray, 
        center: np.ndarray,
        total_sesseion: int = 26,
        overlap_thre: float = OVERLAP_THRE
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
            if i > self._curr_ref_field or i <= self._curr_ref_field - REITERATE_TIMES:
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
        
        if prev_overlap/active_idx.shape[0] < OVERLAP_THRE:
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
        overlap_thre: float = OVERLAP_THRE
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

        if self.maze_type == 0:
            return field_reg, field_info
        
        # Sort with distances to the start points.
        mean_distance = np.nanmean(field_info[:, :, 4], axis=0)
        sort_indices = np.argsort(mean_distance)
        field_reg = field_reg[:, sort_indices]
        field_info = field_info[:, sort_indices, :] 
        return field_reg, field_info

    def _merge_columns(self, matrix):
        return np.apply_along_axis(lambda row: row[~np.isnan(row)][0] if np.any(~np.isnan(row)) else np.nan, 1, matrix)

    def _get_cumulative_rate_map(
        self,
        field_reg: np.ndarray,
        field_info: np.ndarray,
        smooth_map_all: np.ndarray
    ) -> np.ndarray:
        
        cumulative_map = np.zeros((field_reg.shape[1], smooth_map_all.shape[1]))
        for i in range(field_reg.shape[1]):
            for j in range(field_reg.shape[0]):
                if field_reg[j, i] == 1 and np.where(field_info[j, :, 2] == field_info[j, i, 2])[0].shape[0] == 1:
                    cumulative_map[i, self._place_fields[j][int(field_info[j, i, 2])]-1] += smooth_map_all[j, self._place_fields[j][int(field_info[j, i, 2])]-1]

        cumulative_map = cumulative_map.T / np.max(cumulative_map, axis=1)
        cumulative_map = cumulative_map.T
        
        #import matplotlib.pyplot as plt
        #for i in range(cumulative_map.shape[0]):
        #    im = plt.imshow(np.reshape(cumulative_map[i, :], (48, 48)))
        #    plt.colorbar(im)
        #    plt.show()
        return np.where(cumulative_map >= MERGED_OVERLAP_THRE, 1, 0), cumulative_map
        

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
        
        cumulative_map, cumulative_rate_map = self._get_cumulative_rate_map(field_reg, field_info, smooth_map_all)
        merge_matrix = np.zeros((field_reg.shape[1], field_reg.shape[1]))
        for i in range(field_reg.shape[1]):
            for j in range(field_reg.shape[1]):
                if i == j:
                    merge_matrix[i, j] = 1
                    continue
                
                # New method to generate merge_matrix
                if np.where((cumulative_map[i, :] == 1)&(cumulative_map[j, :] == 1))[0].shape[0] >= 1:
                    # The two fields share an overlap greater than MERGED_OVERLAP_THRE
                    merge_matrix[i, j] = merge_matrix[j, i] = 1
                #if np.where(field_info[:, i, 2] - field_info[:, j, 2] == 0)[0].shape[0] >= MERGED_IDENTIFY_THRE: # Merged for two times field
                #    merge_matrix[i, j] = merge_matrix[j, i] = 1

        # Fill the blocks
        for i in range(merge_matrix.shape[0]):
            transition = np.ediff1d(np.concatenate([[0], merge_matrix[i, :], [0]]))
            start_idx = np.where(transition == 1)[0][0]
            end_idx = np.where(transition == -1)[0][-1]
            merge_matrix[start_idx:end_idx, start_idx:end_idx] = 1
                    
        merge_matrix[merge_matrix > 1] = 1
        continue_border = merge_matrix.shape[0]
        merged_place_field = [{} for i in range(field_reg.shape[0])]
        
        for i in range(merge_matrix.shape[0]-1, -1, -1):
            if i >= continue_border:
                continue
            
            transition = np.ediff1d(np.concatenate([[0], merge_matrix[i, :], [0]]))
            if np.where(transition != 0)[0].shape[0] == 0:
                assert False
                continue # No need to merge
            else:
                start_idx = np.where(transition == 1)[0][0]
                end_idx = np.where(transition == -1)[0][-1]
                continue_border = start_idx
                if start_idx == end_idx-1:
                    # No overlaping with other fields
                    for d in range(len(merged_place_field)):
                        if np.isnan(field_info[d, start_idx, 2]) == False:
                            arg_max = np.argmax(cumulative_rate_map[start_idx, :]) + 1
                            field_info[d, start_idx, 2] = arg_max
                            field_info[d, start_idx, 3] = len(cumulative_rate_map[start_idx, :])
                            merged_place_field[d][arg_max] = np.where(cumulative_map[start_idx, :] > 0)[0] + 1
                            # merged_place_field[d][int(field_info[d, start_idx, 2])] = self._place_fields[d][int(field_info[d, start_idx, 2])]
                    continue

                # If there're more than 3 fields merged, remove them to control the quality of data.
                """
                if end_idx-start_idx >= MERGED_FIELD_SUPREME:
                    field_centers = field_info[:, start_idx:end_idx, 2]
                    
                    for j in range(field_centers.shape[0]): 
                        present_centers = np.unique(field_centers[j, np.where(np.isnan(field_centers[j, :]) == False)[0]].astype(int))

                        #for k in present_centers:
                        #    if np.where(field_info[j, :, 2] == k)[0].shape[0] == np.where(field_centers[j, :] == k)[0].shape[0]:
                        #        del self._place_fields[j][k]
                            
                    field_reg = np.delete(field_reg, np.arange(start_idx, end_idx), axis=1)
                    field_info = np.delete(field_info, np.arange(start_idx, end_idx), axis=1)
                    continue
                """                    
                # Merge fields between start_idx and end_idx
                merged_reg = np.sum(field_reg[:, start_idx:end_idx], axis=1)
                merged_reg[merged_reg > 1] = 1
                field_reg = np.delete(field_reg, np.arange(start_idx+1, end_idx), axis=1)
                                       
                field_reg[:, start_idx] = merged_reg
               
                # Update place field all and update field info
                field_centers = field_info[:, start_idx:end_idx, 2]
                
                field_area = np.sum(cumulative_map[start_idx:end_idx, :], axis=0)
                field_area = np.where(field_area > 0)[0] + 1
                arg_max = np.argmax(np.sum(cumulative_rate_map[start_idx:end_idx, :], axis=0)) + 1
                for j in range(field_centers.shape[0]):

                    idx = np.where(np.isnan(field_centers[j, :]) == False)[0]
                    present_centers = np.unique(field_centers[j, np.where(np.isnan(field_centers[j, :]) == False)[0]].astype(int))
                    if len(present_centers) >= 1:
                        field_info[j, start_idx, 2] = arg_max# present_centers[0]
                        field_info[j, start_idx, 3] = len(field_area) # field_info[j, start_idx + idx[0], 3]
                        field_info[j, start_idx, 4] = np.nan# field_info[j, start_idx + idx[0], 4]
                        merged_place_field[j][arg_max] = field_area #self._place_fields[j][present_centers[0]]
                    """ 
                    elif len(present_centers) > 1:
                        #field_area = np.concatenate([self._place_fields[j][k] for k in present_centers])

                        #for k in present_centers:
                        #    if np.where(field_info[j, :, 2] == k)[0].shape[0] == np.where(field_centers[j, :] == k)[0].shape[0]:
                        #        del self._place_fields[j][k]
                            
                        # arg_max = np.argmax(smooth_map_all[j, present_centers-1])+1
                        #self._place_fields[j][present_centers[arg_max]] = field_area
                        field_info[j, start_idx, 2] = present_centers[arg_max]
                        field_info[j, start_idx, 3] = len(field_area)
                        field_info[j, start_idx, 4] = field_info[j, start_idx + idx[arg_max], 4]
                    
                        merged_place_field[j][present_centers[arg_max]] = field_area
                    """      
                
                field_info = np.delete(field_info, np.arange(start_idx+1, end_idx), axis=1)
        
        for d in range(len(merged_place_field)):
            if self._place_fields[d] is None:
                merged_place_field[d] = None
        
        return field_reg, field_info, merged_place_field # self._place_fields

    @staticmethod
    def field_register(
        index_map: np.ndarray,
        place_field_all: list[list[dict]],
        is_placecell: np.ndarray,
        smooth_map_all: np.ndarray,
        overlap_thre: float = OVERLAP_THRE,
        maze_type: int = 1,
        is_shuffle: bool = False
    ) -> tuple[np.ndarray, np.ndarray, list[list[dict]]]:
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
            
            if reg.shape[1] == 0:
                continue
            
            if maze_type != 0 and is_shuffle == False:
                reg, info, place_field_all[i] = neuron._merge_fields(
                    field_reg=reg, 
                    field_info=info,
                    smooth_map_all=smooth_map_all[:, i, :]
                )
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
    overlap_thre: float = OVERLAP_THRE,
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
        smooth_map_all=smooth_map_all,
        is_shuffle = is_shuffle
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
        smooth_map_all=smooth_map_all,
        is_shuffle = is_shuffle
    )
    field_ids = get_field_ids(field_info)
    
    DATA['trs'] = {"is_placecell": is_placecell, "place_field_all": place_field_all, "field_reg": field_reg, "field_info": field_info, 'field_ids': field_ids,
                   "field_centers": get_field_centers(field_info, maze_type)}
    with open(os.path.join(os.path.dirname(cellreg_dir), prefix+appendix+".pkl"), 'wb') as handle:
        print(os.path.join(os.path.dirname(cellreg_dir), prefix+appendix+".pkl"))
        pickle.dump(DATA, handle)
    
def stellar(p_value: float):
    p_value = 1 - np.abs(0.5 - p_value) * 2
    if p_value >= 0.05 or np.isnan(p_value):
        return ""
    elif p_value < 0.05 and p_value >= 0.01:
        return "*"
    elif p_value < 0.01 and p_value >= 0.001:
        return "**"
    elif p_value < 0.001 and p_value >= 0.0001:
        return "***"
    elif p_value < 0.0001:
        return "****"
    
import copy as cp
from mazepy.datastruc.neuact import SpikeTrain
from mazepy.datastruc.variables import Variable1D
class TrackerDsp(object):
    def __init__(self):
        pass
    
    def register(
        self,
        init_rate: np.ndarray,
        field_area: np.ndarray,
        Spikes: np.ndarray,
        spike_nodes: np.ndarray,
        occu_time: np.ndarray,
        Ms: np.ndarray,
        n_shuffle: int = 10000,
        percent: int | float = 95,
        is_return_shuffle: bool = False
    ):
        assert len(init_rate) == 10
        assert len(Spikes) == 10
        assert len(spike_nodes) == 10
        assert len(occu_time) == 10
        
        idxs = [
            np.where(np.isin(spike_nodes[i], field_area))[0] for i in range(10)
        ]
        
        n_spikes = np.sum(
            [np.sum(Spikes[i][idxs[i]]) for i in range(10)]
        )
        
        n_frames = np.sum(
            [idxs[i].shape[0] for i in range(10)]
        )
        
        rig_boundary = np.cumsum(
            [idxs[i].shape[0] for i in range(10)]
        )
        lef_boundary = np.concatenate([[0], rig_boundary[:-1]])
        
        rates = np.zeros((n_shuffle, 10), np.float64)
        
        Spikes_rand = [[] for _ in range(10)]

        for i in range(n_shuffle):
            Spikes_r = self.shuffle_spikes(
                Spikes=cp.deepcopy(Spikes),
                idxs=idxs,
                n_spikes=n_spikes,
                n_frames=n_frames,
                lef_boundary=lef_boundary,
                rig_boundary=rig_boundary
            )
            for j in range(10):
                Spikes_rand[j].append(Spikes_r[j][idxs[j]])
                
        for j in range(10):
            Spikes_rand[j] = np.vstack(Spikes_rand[j])
            if Spikes_rand[j].shape[1] == 0:
                continue

            try:
                res = _convert_to_kilosort_form(Spikes_rand[j])
                kilosort_spikes = res[0, :]
                kilosort_variables = spike_nodes[j][idxs[j]] - 1
                spike_counts = _get_kilosort_spike_counts(
                    kilosort_spikes.astype(np.int64),
                    kilosort_variables.astype(np.int64),
                    2304
                ) 

                rate = spike_counts/(occu_time[j]/1000)
                rate[np.isnan(rate) | np.isinf(rate)] = 0
                smooth_rate = rate @ Ms.T
                rates[:, j] = np.max(smooth_rate[:, field_area-1], axis=1)
            except:
                smooth_rate = calc_ratemap(
                    Spikes=Spikes_rand[j],
                    spike_nodes=spike_nodes[j][idxs[j]],
                    occu_time=occu_time[j],
                    Ms=Ms
                )[2]
             
                rates[:, j] = np.max(smooth_rate[:, field_area-1], axis=1)
    
        thre = np.percentile(rates, 100-percent/2, axis=0)
        
        reg = np.where((init_rate - thre >= 0), 1., 0.)
        
        for i in range(10):
            if idxs[i].shape[0] <= 10:
                reg[i] = np.nan
                rates[:, i] = np.nan
                thre[i] = np.nan
        
        P = np.array([np.where(rates[:, i] < init_rate[i])[0].shape[0] / n_shuffle for i in range(10)])
        P += reg * 0 # Add nan
        is_high_quality = np.where(thre - 0.2 < 0)[0].shape[0] == 0
        
        if is_return_shuffle:
            return reg, P, thre, is_high_quality, rates
        else:
            return reg, P, thre, is_high_quality
        
    def shuffle_spikes(
        self,
        Spikes: np.ndarray,
        idxs: list[np.ndarray],
        n_spikes: int,
        n_frames: int,
        lef_boundary: np.ndarray,
        rig_boundary: np.ndarray
    ) -> np.ndarray:
        spike_seq = np.zeros(n_frames)
        spike_seq[np.random.choice(n_frames, n_spikes, replace=False)] = 1
        
        for i in range(10):
            Spikes[i][idxs[i]] = 0
            Spikes[i][idxs[i]] = spike_seq[lef_boundary[i]:rig_boundary[i]]
        
        return Spikes

    @staticmethod
    def visualize_single_field(
        trace, 
        cell: int,
        field_center: int,
        save_loc: str,
        file_name: str | None = None,
        n_shuffle: int = 10000,
        percent: int | float = 95
    ):
        if os.path.exists(save_loc) == False:
            os.mkdir(save_loc)
            
        if field_center not in trace['place_field_all'][cell].keys():
            raise ValueError(
                f"Field {field_center} not found in cell {cell}."
                f"Only {list(trace['place_field_all'][cell].keys())} are available."
            )

        Spikes = cp.deepcopy([
            trace[f'node {i}']['Spikes'] for i in range(10)
        ])
        
        spike_nodes = [
            trace[f'node {i}']['spike_nodes'] for i in range(10)
        ]
        
        occu_time = cp.deepcopy([
            trace[f'node {i}']['occu_time_spf'] for i in range(10)
        ])
        
        field_area = trace['place_field_all'][cell][field_center]
        init_rate = np.array([
                np.max(trace[f'node {j}']['smooth_map_all'][cell, field_area-1]) for j in range(10)
            ])
        tracker = TrackerDsp()
        reg, P, thre, is_hq, shuf_rate = tracker.register(
            init_rate=init_rate,
            field_area=field_area,
            Spikes=[Spikes[j][cell, :] for j in range(10)],
            spike_nodes=cp.deepcopy(spike_nodes),
            occu_time=occu_time,
            n_shuffle=n_shuffle, 
            Ms=trace['Ms'],
            is_return_shuffle=True
        )
        print(f"Reg: {reg}\n  P-values: {P}\n  Threshold: {thre}")
        v_max = max(np.nanmax(shuf_rate), np.nanmax(init_rate))
        v_max = max(1, int(v_max) + 1)
        min_thre, max_thre = np.nanpercentile(shuf_rate, [(100 - percent)/2, 100 - (100 - percent)/2], axis=0)
        print(f"  Min Threshold: {min_thre}; Max Threshold: {max_thre}")

        # Visualize Radar Chart.
        angles = np.linspace(0, 2*np.pi, 10, endpoint=False)
        
        min_thre = np.append(min_thre, min_thre[0])
        max_thre = np.append(max_thre, max_thre[0])
        angles = np.append(angles, angles[0])
        degree = 2 * np.pi / 10
        init_rate = np.append(init_rate, init_rate[0])
        
        fig = plt.figure(figsize=(4, 4))
        ax = plt.subplot(111, projection='polar')
        
        colors = ['#A9CCE3', '#A8DADC', '#9C8FBC', '#D9A6A9', '#A9CCE3',
                  '#A9CCE3', '#F2E2C5', '#647D91', '#C06C84', '#A9CCE3']
        
        idx = np.where(np.isnan(max_thre) == False)[0]
        nan_idx = np.where(np.isnan(max_thre) == True)[0]
        angle_lef = angles - degree / 2
        angle_rig = angles + degree / 2
        ax.fill(angles[idx], max_thre[idx], color='grey', alpha=0.3, edgecolor = None)
        ax.fill(angles[idx], min_thre[idx], color='white', edgecolor = None)
        for nidx in nan_idx:
            ax.fill([angle_lef[nidx], angle_rig[nidx],  angle_lef[nidx]], [v_max, v_max, 0], color='white',edgecolor = None)
        ax.plot(angles, init_rate, color = 'k', linewidth = 0.5)
            
        ax.set_aspect('equal')
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        for i in range(len(angles)-1):
            ax.text(angles[i], v_max, stellar(P[i]), ha='center', va='center', color = 'k', fontsize = 4)
        
        labels = ['1a', '2', '3', '4', '1b', '1c', '5', '6', '7', '1d']
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    
        ax.spines['polar'].set_linewidth(0.5)
        
        for i, line in enumerate(ax.yaxis.get_gridlines()):
            line.set_linewidth(0.5)
        
        for i, line in enumerate(ax.xaxis.get_gridlines()):
            line.set_linewidth(0.5)
            line.set_color(colors[i])
    
        ax.set_ylim(0, v_max)    
        if file_name is None:
            file_name = f'Cell_{cell+1}_Field_{field_center}'
        
        color = "red" if is_hq else "black"
        ax.set_title(file_name, color=color)
        
        plt.savefig(os.path.join(save_loc, file_name + '.png'), dpi=600)
        plt.savefig(os.path.join(save_loc, file_name + '.svg'))
        plt.close()
    
    @staticmethod
    def field_register(trace, n_shuffle=10000, percent = 99) -> tuple[np.ndarray, np.ndarray]:
        field_reg = []
        field_info = []
        is_high_quality = []
        
        Spikes = cp.deepcopy([
            trace[f'node {i}']['Spikes'] for i in range(10)
        ])
        
        spike_nodes = [
            trace[f'node {i}']['spike_nodes'] for i in range(10)
        ]
        
        occu_time = cp.deepcopy([
            trace[f'node {i}']['occu_time_spf'] for i in range(10)
        ])
        
        qualified_cells = np.arange(trace['n_neuron'])
        
        delete_keys = [[] for _ in range(trace['n_neuron'])]
        
        shuf_rate_accu = []
        
        for i in tqdm(qualified_cells):
            for k in trace['place_field_all'][i].keys():
                tracker = TrackerDsp()
                field_area = trace['place_field_all'][i][k]
                reg, P, thre, is_hq, shuf_rate = tracker.register(
                    init_rate=np.array([
                        np.max(trace[f'node {j}']['smooth_map_all'][i, field_area-1]) for j in range(10)
                    ]),
                    field_area=field_area,
                    Spikes=[Spikes[j][i, :] for j in range(10)],
                    spike_nodes=cp.deepcopy(spike_nodes),
                    occu_time=occu_time,
                    n_shuffle=n_shuffle, 
                    Ms=trace['Ms'],
                    is_return_shuffle=True,
                    percent=percent
                )
                
                if np.nansum(reg) == 0:
                    delete_keys[i].append(k)
                else:
                    info = np.full((10, 7), np.nan)
                    info[:, 0] = i+1
                    info[:, 1] = np.array([
                        trace[f'node {j}']['is_placecell'][i] for j in range(10)
                    ])
                    info[:, 2] = k
                    info[:, 3] = len(field_area)
                    info[:, 4] = P
                    info[:, 5] = 1 if is_hq else 0
                    info[:, 6] = thre
                    
                    field_reg.append(reg)
                    field_info.append(info)
                    shuf_rate_accu.append(shuf_rate)
                    
        for i in qualified_cells:
            for k in delete_keys[i]:
                trace['place_field_all'][i].pop(k)
        
        field_reg = np.vstack(field_reg).T
        info = np.zeros((field_reg.shape[0], field_reg.shape[1], 7), np.float64)
        for i in range(field_reg.shape[1]):
            info[:, i, :] = field_info[i]
        
        shuf_rate_all = np.zeros((field_reg.shape[1], n_shuffle, 10), np.float64)
        for i in range(field_reg.shape[1]):
            shuf_rate_all[i, :, :] = shuf_rate_accu[i]
        
        if os.path.exists(trace['p']) == False:
            os.mkdir(trace['p'])
            
        with open(os.path.join(trace['p'], "field_shuffle.pkl"), 'wb') as handle:
            pickle.dump(shuf_rate_all, handle)
        return field_reg, info
    
if __name__ == '__main__':
    from mylib.local_path import f2
    import pickle
    
    for i in range(len(f2)):
        with open(f2['Trace File'][i], 'rb') as handle:
            trace = pickle.load(handle)
        
        tracker = TrackerDsp.field_register(
            trace,
            n_shuffle=10000
        )
        
        with open(f2['Trace File'][i], 'wb') as handle:
            pickle.dump(trace, handle)