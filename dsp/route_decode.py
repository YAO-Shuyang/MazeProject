from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
from tqdm import tqdm

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
    
    # cross validation for n_cross_times
    combinations = _cross_validation(route_ids, lap_ids, n_times=n_cross_times)

    y_test_all, y_pred_all = [], []
    x_test_pos = []
    for i in tqdm(range(combinations.shape[0])):
        X_train, X_test, y_train, y_test, test_id = _split_train_test(
            neural_traj, 
            pos_traj, 
            route_ids, 
            lap_ids, 
            combinations[i, :]
        )
        y_pred = predict_routes(X_train, X_test, y_train, max_iter)
        y_pred_all.append(y_pred)
        y_test_all.append(y_test)
        x_test_pos.append(pos_traj[test_id])
    
    return np.concatenate(y_pred_all), np.concatenate(y_test_all), np.concatenate(x_test_pos)

def _cross_validation(
    route_ids: np.ndarray,
    lap_ids: np.ndarray,
    n_times: int = 100
) -> np.ndarray:
    lap_intervals = np.where(np.ediff1d(lap_ids) != 0)[0] + 1
    lap_beg = np.concatenate([[0], lap_intervals])
    lap_end = np.concatenate([lap_intervals, [lap_ids.shape[0]]])
    lap_ids = lap_ids[lap_beg]
    lap_routes = route_ids[lap_beg]
    
    idx_route0 = lap_ids[np.where(lap_routes == 0)[0]]
    idx_route1 = lap_ids[np.where(lap_routes == 1)[0]]
    idx_route2 = lap_ids[np.where(lap_routes == 2)[0]]
    idx_route3 = lap_ids[np.where(lap_routes == 3)[0]]
    idx_route4 = lap_ids[np.where(lap_routes == 4)[0]]
    idx_route5 = lap_ids[np.where(lap_routes == 5)[0]]
    idx_route6 = lap_ids[np.where(lap_routes == 6)[0]]
    
    max_times = idx_route0.shape[0] * idx_route1.shape[0] * idx_route2.shape[0] * idx_route3.shape[0] * idx_route4.shape[0] * idx_route5.shape[0] * idx_route6.shape[0]
    if n_times > max_times:
        warnings.warn(
            f"The number of times to cross validation ({n_times}) is too large. "
            f"The number of combinations will be too large. Please reduce the "
            f"number of times to cross validation. The maximum number of validation "
            f"is {max_times}. We suggested not to use more than 1000 times."
        )
        
    n_times = min(n_times, max_times)
    
    combinations = np.zeros((n_times, 7))
    temps = []
    i = 0
    while i < n_times:
        comb_rand = (
            np.random.choice(idx_route0),
            np.random.choice(idx_route1),
            np.random.choice(idx_route2),
            np.random.choice(idx_route3),
            np.random.choice(idx_route4),   
            np.random.choice(idx_route5),
            np.random.choice(idx_route6)
        )
        
        if comb_rand not in temps:
            combinations[i, :] = np.asarray(comb_rand)
            temps.append(comb_rand)
            i += 1

    return combinations

def _split_train_test(
    neural_traj: np.ndarray, 
    pos_traj: np.ndarray, 
    route_ids: np.ndarray,
    lap_ids: np.ndarray,
    combinations: np.ndarray
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
    X = np.vstack([neural_traj, pos_traj]).T
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    test_idx = np.concatenate(
        [np.where(lap_ids == comb)[0] for comb in combinations]
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
    
    with open(r"E:\Data\Dsp_maze\10224\20231014\trace.pkl", 'rb') as handle:
        trace = pickle.load(handle)
        neural_traj = trace['neural_traj']
        pos_traj = trace['pos_traj']
        lap_ids = trace['traj_lap_ids']
        route_ids = trace['traj_route_ids']
        combinations = _cross_validation(route_ids, lap_ids, n_times=1)
        print("Cross Validation Done.")
        X_train, X_test, y_train, y_test = _split_train_test(neural_traj, pos_traj, route_ids, lap_ids, combinations[0, :])
        print("Split Data Done.")
        t1 = time.time()
        y_pred = predict_routes(X_train, X_test, y_train)
        print("Time cost:", time.time() - t1)
        print(y_pred)
