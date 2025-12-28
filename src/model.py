from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

class MonitoredGBR(GradientBoostingRegressor):
    def fit(self, X, y, sample_weight=None, monitor=None):
        return super().fit(X, y, sample_weight=sample_weight, monitor=self._monitor_callback)
        
    def _monitor_callback(self, i, estimator, locals):
        if i % 10 == 0 or i == self.n_estimators - 1:
            train_loss = estimator.train_score_[i]
            if hasattr(estimator, 'validation_score_'):
                # validation_score_ is populated if validation_fraction > 0
                val_loss = estimator.validation_score_[i]
                print(f"Iter {i+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            else:
                print(f"Iter {i+1}: Train Loss = {train_loss:.4f}")
        return False

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
    gb_regressor = MonitoredGBR(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=0
    )
    
    model = MultiOutputRegressor(gb_regressor)
    
    return model
