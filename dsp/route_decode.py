from turtle import pos
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
from tqdm import tqdm
from mylib.maze_graph import CorrectPath_maze_1 as CP
from mylib.maze_utils3 import GetDMatrices
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def compute_accuracy(y_pred, y_test, x_test):
    accuracy = np.zeros((7, len(CP)), dtype=np.float64)
    
    for j in range(7):
        for i in range(len(CP)):
            n_all = np.where(
                (x_test == CP[i]) & (y_test == j)
            )[0].shape[0]
            n_correct = np.where(
                (y_test == j) &
                (y_pred == j) & 
                (x_test == CP[i])
            )[0].shape[0]
            if n_all == 0:
                accuracy[j, i] = np.nan
            else:
                accuracy[j, i] = n_correct / n_all
        
    return accuracy

def decode_routes(
    trace: dict, 
    n_cross_times: int = 100,
    max_iter: int = 10000
) -> tuple[np.ndarray]:
    """
    Decode the route of each lap given the neural trajectory and position.

    Parameters
    ----------
    trace : dict
        The dictionary of the trace.
    n_cross_times : int, optional
        The number of cross validation, by default 100

    Returns
    -------
    route_preds : np.ndarray
        The predicted route of each lap.
    route_tests : np.ndarray
        The true route of each lap.
    pos_tests : np.ndarray
        The position of each time bin.
    """
    neural_traj = trace['neural_traj']
    pos_traj = trace['pos_traj']
    lap_ids = trace['traj_lap_ids']
    route_ids = trace['traj_route_ids']
    segment_ids = trace['traj_segment_ids']
    
    D = GetDMatrices(1, 48)
    #dis_traj = D[pos_traj-1, 0]
    
    x, y = (pos_traj-1) // 48, (pos_traj-1) % 48

    y_test_all, y_pred_all = [], []
    x_test_pos = []
    decode_type = []
    segments = []
    
    for seg in range(0, 7):
        indices = np.where(
            segment_ids == seg
        )[0]
        # cross validation for n_cross_times
        combinations = _cross_validation(
            route_ids[indices], 
            lap_ids[indices], 
            seg=seg,
            n_times=n_cross_times
        )
        
        for i in tqdm(range(combinations.shape[0])):
            X_train, X_test, y_train, y_test, test_id = _split_train_test(
                neural_traj[:, indices],
                route_ids[indices], 
                lap_ids[indices], 
                combinations[i, :],
                seg=seg
            )
            
            if seg >= 1:
                n_components = seg
                lda = LDA(n_components=n_components)
                X_train = lda.fit_transform(
                    X_train, 
                    y_train
                )
                X_test = lda.transform(X_test)
                
            print(seg, np.unique(y_train), np.unique(y_test))
                        
            y_pred = predict_routes(X_train, X_test, y_train, max_iter)
            y_pred_all.append(y_pred)
            y_test_all.append(y_test)
            x_test_pos.append(pos_traj[indices][test_id])
            segments.append(np.repeat(seg, y_pred.shape[0]))
        
            decode_type.append(np.repeat(0, y_test.shape[0]))

            for s in range(10):
                y_train = y_train[np.random.permutation(y_train.shape[0])]
            
            y_pred = predict_routes(X_train, X_test, y_train, max_iter)
            y_pred_all.append(y_pred)
            y_test_all.append(y_test)
            x_test_pos.append(pos_traj[indices][test_id])
            segments.append(np.repeat(seg, y_pred.shape[0]))
        
            decode_type.append(np.repeat(1, y_test.shape[0]))
              
    return np.concatenate(y_pred_all), np.concatenate(y_test_all), np.concatenate(x_test_pos), np.concatenate(decode_type), np.concatenate(segments)

def _cross_validation(
    route_ids: np.ndarray,
    lap_ids: np.ndarray,
    seg: int,
    n_times: int = 100
) -> np.ndarray:
    lap_intervals = np.where(np.ediff1d(lap_ids) != 0)[0] + 1
    lap_beg = np.concatenate([[0], lap_intervals])
    lap_end = np.concatenate([lap_intervals, [lap_ids.shape[0]]])
    lap_ids = lap_ids[lap_beg]
    lap_routes = route_ids[lap_beg]
    
    segmented_routes = [0, 4, 1, 5, 2, 6, 3]
    idx_routes = [
        lap_ids[np.where(lap_routes == segmented_routes[i])[0]] for i in range(seg+1)
    ]
    n_lens = np.array([
        idx_route.shape[0] for idx_route in idx_routes
    ])
    
    max_times = np.cumprod(n_lens)[-1]
    if n_times > max_times:
        warnings.warn(
            f"The number of times to cross validation ({n_times}) is too large. "
            f"The number of combinations will be too large. Please reduce the "
            f"number of times to cross validation. The maximum number of validation "
            f"is {max_times}. We suggested not to use more than 1000 times."
        )
        
    n_times = min(n_times, max_times)
    
    combinations = np.zeros((n_times, 7)) * np.nan
    temps = []
    i = 0
    while i < n_times:
        comb_rand = [
            np.random.choice(idx_route) for idx_route in idx_routes
        ]
        
        if comb_rand not in temps:
            combinations[i, :seg+1] = np.asarray(comb_rand)
            temps.append(comb_rand)
            i += 1

    return combinations

def _split_train_test(
    neural_traj: np.ndarray,
    route_ids: np.ndarray,
    lap_ids: np.ndarray,
    combinations: np.ndarray,
    seg: int
):
    """
    Split the data into training and testing sets.
    
    Parameters
    ----------
    neural_traj : numpy.ndarray
        An (n, T) matrix where n is the number of neurons and T is the number of time bins.
    pos_traj : numpy.ndarray
        A T-dim vector indicating the position at each time bin.
    route_ids : numpy.ndarray
        A T-dim vector indicating the route ID (0-6) at each time bin.
    lap_ids : numpy.ndarray
        A T-dim vector indicating the lap ID at each time bin.
    combinations : numpy.ndarray
        An (7, ) array selecting one lap for each routes as the test set.
        
    Returns
    -------
    X_train : numpy.ndarray
        An (n_train, T) matrix where n_train is the number of training samples and T is the number of time bins.
    X_test : numpy.ndarray
        An (n_test, T) matrix where n_test is the number of testing samples and T is the number of time bins.
    y_train : numpy.ndarray
        A T-dim vector indicating the route ID (0-6) at each time bin in the training set.
    y_test : numpy.ndarray
        A T-dim vector indicating the route ID (0-6) at each time bin in the testing set.
    """
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(neural_traj.T)

    # Split the data into training and testing sets
    test_idx = np.concatenate(
        [np.where(lap_ids == comb)[0] for comb in combinations[:seg+1].astype(np.int64)]
    )
    train_idx = np.setdiff1d(np.arange(X.shape[0]), test_idx)

    return X[train_idx, :], X[test_idx, :], route_ids[train_idx], route_ids[test_idx], test_idx

def predict_routes(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = 10000
):
    """
    Predict routes based on neuronal population activity and position.
    """
    # Initialize the SVM classifier with the One-vs-Rest strategy
    #svm_classifier = OneVsRestClassifier(
    #    SVC(kernel='linear', max_iter=max_iter)
    #)
    svm_classifier = OneVsRestClassifier(
        SVC(kernel='rbf', gamma='scale', max_iter=max_iter)
    )
    # Train the classifier
    svm_classifier.fit(X_train, y_train)

    # Predict on the test set
    y_pred = svm_classifier.predict(X_test)
    return y_pred


if __name__ == '__main__':
    import pickle
    import time 
    import os
    from mylib.local_path import f2
    
    for i in range(21, 28):
        print(i, f2['MiceID'][i], f2['date'][i], ' session '+str(f2['session'][i]))
        with open(f2['Trace File'][i], 'rb') as handle:
            trace = pickle.load(handle)
        
        y_pred, y_test, x_test, decode_type, segments = decode_routes(trace, n_cross_times=50, max_iter=100000)
        #y_pred_ctrl, y_test_ctrl, x_test_ctrl = decode_routes(trace, n_cross_times=10, is_control=True)
        
        with open(os.path.join(f2['Path'][i], "route_decode.pkl"), 'wb') as handle:
            pickle.dump(
                {
                    "y_pred": y_pred,
                    "y_test": y_test,
                    "x_test": x_test,
                    "decode_type": decode_type,
                    "segments": segments
                }, 
                handle
            )