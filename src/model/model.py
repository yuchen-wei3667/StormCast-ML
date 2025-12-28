from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor



class MonitoredGBR(GradientBoostingRegressor):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=5, 
                 random_state=42, validation_fraction=0.1, n_iter_no_change=10,
                 verbose=0, X_val=None, y_val=None, **kwargs):
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            verbose=verbose,
            **kwargs
        )
        self.X_val = X_val
        self.y_val = y_val
        self.val_scores_ = []
        
    def fit(self, X, y, sample_weight=None, monitor=None):
        return super().fit(X, y, sample_weight=sample_weight, monitor=self._monitor_callback)
        
    def _monitor_callback(self, i, estimator, locals):
        """Monitor callback that computes and displays validation loss"""
        if i % 10 == 0 or i == self.n_estimators - 1:
            train_loss = self.train_score_[i]
            
            # Compute validation loss if validation data provided
            if self.X_val is not None and self.y_val is not None:
                # Use staged_predict to get predictions at this iteration
                y_pred = None
                for j, pred in enumerate(self.staged_predict(self.X_val)):
                    if j == i:
                        y_pred = pred
                        break
                
                if y_pred is not None:
                    from sklearn.metrics import mean_squared_error
                    val_loss = mean_squared_error(self.y_val, y_pred)
                    self.val_scores_.append(val_loss)
                    print(f"Iter {i+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                else:
                    print(f"Iter {i+1}: Train Loss = {train_loss:.4f}")
            else:
                print(f"Iter {i+1}: Train Loss = {train_loss:.4f}")
        
        # Dynamic early stopping based on validation loss
        if self.X_val is not None and len(self.val_scores_) > self.n_iter_no_change:
            recent_val_scores = self.val_scores_[-self.n_iter_no_change:]
            if all(recent_val_scores[i] >= recent_val_scores[0] for i in range(1, len(recent_val_scores))):
                print(f"Early stopping at iteration {i+1}: validation loss not improving")
                return True  # Stop training
        
        return False


def create_gb_model(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, X_val=None, y_val=None):
    """
    Creates a MultiOutput Gradient Boosting Regressor model.
    
    Args:
        n_estimators (int): The number of boosting stages to perform.
        learning_rate (float): Learning rate shrinks the contribution of each tree.
        max_depth (int): Maximum depth of the individual regression estimators.
        random_state (int): Controls the random seed.
        X_val (array): Validation features for monitoring (optional).
        y_val (array): Validation targets for monitoring (optional).
        
    Returns:
        model: A MultiOutputRegressor wrapping GradientBoostingRegressor.
    """
    # Create separate estimators for each output to pass validation data
    estimators = []
    for i in range(2):  # u and v velocity
        y_val_single = y_val[:, i] if y_val is not None else None
        gb = MonitoredGBR(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            n_iter_no_change=10,
            verbose=0,
            X_val=X_val,
            y_val=y_val_single
        )
        estimators.append(gb)
    
    # Manually create MultiOutputRegressor with our custom estimators
    from sklearn.multioutput import MultiOutputRegressor
    model = MultiOutputRegressor(estimators[0])
    model.estimators_ = estimators  # Override with our pre-configured estimators
    
    return model
