"""
Improved Gradient Boosting Model with XGBoost and better hyperparameters
"""
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, falling back to sklearn GradientBoostingRegressor")


class MonitoredGBR(GradientBoostingRegressor):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=5, 
                 random_state=42, validation_fraction=0.1, n_iter_no_change=10,
                 verbose=0, X_val=None, y_val=None, subsample=0.8, **kwargs):
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            verbose=verbose,
            subsample=subsample,
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


class MonitoredXGB(xgb.XGBRegressor):
    """XGBoost wrapper that inherits from XGBRegressor for sklearn compatibility"""
    def __init__(self, n_estimators=300, learning_rate=0.05, max_depth=5,
                 random_state=42, X_val=None, y_val=None, subsample=0.8, **kwargs):
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            subsample=subsample,
            colsample_bytree=0.8,
            **kwargs
        )
        self.X_val = X_val
        self.y_val = y_val
        
    def fit(self, X, y, **kwargs):
        """Fit XGBoost model with validation monitoring"""
        if self.X_val is not None and self.y_val is not None:
            eval_set = [(X, y), (self.X_val, self.y_val)]
            return super().fit(X, y, eval_set=eval_set, verbose=True)
        else:
            return super().fit(X, y, verbose=True)


def create_gb_model(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42, 
                    X_val=None, y_val=None, use_xgboost=True):
    """
    Creates an improved MultiOutput Gradient Boosting Regressor model.
    
    Args:
        n_estimators (int): The number of boosting stages (increased from 100 to 300).
        learning_rate (float): Learning rate (decreased from 0.1 to 0.05 for better precision).
        max_depth (int): Maximum depth of the individual regression estimators.
        random_state (int): Controls the random seed.
        X_val (array): Validation features for monitoring (optional).
        y_val (array): Validation targets for monitoring (optional).
        use_xgboost (bool): Use XGBoost if available (default True).
        
    Returns:
        model: A MultiOutputRegressor wrapping the chosen regressor.
    """
    # Create separate estimators for each output to pass validation data
    estimators = []
    
    # Decide which model to use
    if use_xgboost and XGBOOST_AVAILABLE:
        print("Using XGBoost for improved performance")
        ModelClass = MonitoredXGB
    else:
        print("Using sklearn GradientBoostingRegressor")
        ModelClass = MonitoredGBR
    
    for i in range(2):  # u and v velocity
        y_val_single = y_val[:, i] if y_val is not None else None
        model = ModelClass(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            n_iter_no_change=10,
            verbose=0,
            X_val=X_val,
            y_val=y_val_single,
            subsample=0.8  # Use 80% of data per tree
        )
        estimators.append(model)
    
    # Manually create MultiOutputRegressor with our custom estimators
    from sklearn.multioutput import MultiOutputRegressor
    model = MultiOutputRegressor(estimators[0])
    model.estimators_ = estimators  # Override with our pre-configured estimators
    
    return model
