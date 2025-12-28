from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

def create_gb_model(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42):
    """
    Creates a MultiOutput Gradient Boosting Regressor model.
    
    Args:
        n_estimators (int): The number of boosting stages to perform.
        learning_rate (float): Learning rate shrinks the contribution of each tree.
        max_depth (int): Maximum depth of the individual regression estimators.
        random_state (int): Controls the random seed.
        
    Returns:
        model: A MultiOutputRegressor wrapping GradientBoostingRegressor.
    """
    gb_regressor = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    
    model = MultiOutputRegressor(gb_regressor)
    
    return model
