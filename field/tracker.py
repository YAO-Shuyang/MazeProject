import numpy as np
import warnings


class Field(object):
    def __init__(
        self, 
        curr_session: int, 
        area: np.ndarray, 
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
    def is_active(self):
        return self._is_active
        
    def update(self, curr_session: int, place_field: dict | None) -> bool:
        self._curr_session = curr_session
        
        # If the neuron is not detected, the field is natually no detected subsequently.
        if place_field is None:
            self._area.append(None)
            return False
        
        # Run for all field candidates to identify if they were the same one as this field.
        for k in place_field.keys():
            if self._is_samefield(place_field[k]):
                self._is_active[curr_session] = 1
                self._area.append(place_field[k])
                return True
            
        self._area.append(None)
            
        return False
        
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
            raise ZeroDivisionError(f"There's no active history of this field, indicating 
                                    somewhere is wrong.")
        
        # Criterion 1: Have 60% or more overlapping with the most recently detected fields
        if self._is_overlap(self._area[active_idx[-1]], area, self._thre) == False:
            return False
        
        # Criterion 2: It should have overlap with 50% or more the remaining detected fields.
        prev_overlap = 1
        for i in range(len(active_idx)-1):
            if np.intersect1d(self._area[active_idx[i]], area).shape[0] > 0:
                prev_overlap += 1
        
        if prev_overlap/active_idx.shape[0] < 0.5:
            return False
        else:  
            return True
        
    def register(self, is_nodetect):
        if self.curr_session != self._total_session-1:
            warnings.warn(
                f"Current session is {self.curr_session+1}, but total sessions is {self._total_session}."
            )
        return self.is_active + is_nodetect
        

class Tracker(object):
    def __init__(
        self, 
        curr_session: int, 
        place_fields: list[dict], 
        smooth_maps: np.ndarray,
        is_placecell: np.ndarray,
        total_sesseion: int = 26,
        overlap_thre: float = 0.6
    ) -> None:
        """__init__: Initialize Tracker

        Parameters
        ----------
        curr_session : int
            The current session
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
        assert smooth_maps.shape[0] == total_sesseion
        assert is_placecell.shape[0] == total_sesseion
        assert len(place_fields) == total_sesseion
        
        self._curr_session = curr_session
        self._total_sesseion = total_sesseion
        self._place_fields = place_fields
        self._smooth_maps = smooth_maps
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
                                                     area=self._place_fields[k], 
                                                     total_sesseion=self._total_sesseion, 
                                                     overlap_thre=self._thre))
                break
            
        for i in range(start_session + 1, self._total_sesseion):
            for pf in self._indept_fields:
                pf.update(i, self._place_fields[i])
            
                
            