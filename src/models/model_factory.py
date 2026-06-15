from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


def create_models():
    """Create all baseline models"""
    return {
        "linear_regression_baseline": LinearRegression(),
        "decision_tree_baseline": DecisionTreeRegressor(
            random_state=42,
        ),
        "random_forest_baseline": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
    }
