# EPICS: Agricultural ML Application

## Project Overview
Flask-based web application providing two ML services: **Crop Recommendation** (classification) and **Yield Prediction** (regression). Uses pre-trained scikit-learn models served via Flask Blueprints.

## Architecture

### Module Structure (Blueprint Pattern)
- **Main App** (`app.py`): Minimal Flask app registering two blueprints with URL prefixes
  - `/crop/*` → `crop_recommendation.app.crop_recommendation_app`
  - `/yield/*` → `yield_prediction.app.yield_prediction_app`
- Each module (`crop_recommendation/`, `yield_prediction/`) contains:
  - `app.py`: Blueprint with routes and prediction logic
  - `model_train.py`: Standalone training script that outputs `.pkl` files to `Data/`
  - `__init__.py`: Empty (modules imported directly, not as packages)

### Model Persistence Convention
All models stored in `Data/` directory as pickle files:
- **Crop Recommendation**: `crop_model.pkl` (RandomForest), `label_encoder.pkl` (sklearn LabelEncoder)
- **Yield Prediction**: `dtr.pkl` (DecisionTreeRegressor), `preprocessor.pkl` (ColumnTransformer with OneHotEncoder + StandardScaler)

Models are loaded at module import time with try/except blocks that print errors but don't crash the app.

## Critical Data Flow

### Crop Recommendation
**Input**: 7 soil/weather features (Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH_Value, Rainfall)  
**Process**: Direct prediction → LabelEncoder inverse_transform  
**Output**: Crop name string (e.g., "Rice", "Wheat")

### Yield Prediction
**Input**: 8 features (Crop, Crop_Year, Season, State, Area, Annual_Rainfall, Fertilizer, Pesticide)  
**Process**: 
1. Create DataFrame with exact column order matching training
2. Apply `preprocessor` (OneHot encodes categorical, scales numeric)
3. Predict with `dtr` model
**Output**: Numeric yield value formatted to 2 decimals

**Critical**: Feature order matters! DataFrame columns must match training order: `['Crop', 'Crop_Year', 'Season', 'State', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']`

## Developer Workflows

### Training Models
Models must be retrained manually before first run or after data changes:
```powershell
# Train crop recommendation model
python crop_recommendation/model_train.py  # Outputs to Data/crop_model.pkl, Data/label_encoder.pkl

# Train yield prediction model
python yield_prediction/model_train.py  # Outputs to Data/dtr.pkl, Data/preprocessor.pkl
```

### Running the Application
```powershell
python app.py  # Starts Flask dev server on localhost:5000
```

### Testing
Integration tests in `test_app.py` require server running separately:
```powershell
# Terminal 1: Start server
python app.py

# Terminal 2: Run tests
python test_app.py
```

Tests use `requests` library to hit actual endpoints with sample data. No pytest/unittest framework.

### Debugging Models
Use `debug_models.py` to verify all pickle files load correctly without starting the server.

## Project-Specific Patterns

### Template Sharing
All blueprints share templates via `template_folder='../templates'` and `static_folder='../static'` in blueprint creation. Templates reference parent routes for navigation.

### Error Handling
- Model loading errors are printed but don't prevent app startup (defensive design for development)
- Prediction errors return rendered template with `error` parameter
- No custom exception classes; uses standard Python exceptions

### Path Conventions
- Model training scripts use relative paths to `Data/` from module directory
- Main app uses absolute imports: `from crop_recommendation.app import crop_recommendation_app`
- Data files referenced as `Data/filename.pkl` (relative from project root when app runs)

## External Dependencies
- **Flask**: Web framework
- **pandas**: DataFrame operations for feature preparation
- **scikit-learn**: All ML models (RandomForest, DecisionTreeRegressor) and preprocessing
- **pickle**: Model serialization (standard library)

No database, no API keys, no external services. All models run locally.

## Known Constraints
- Models loaded at import time mean server restart required after retraining
- No model versioning or A/B testing infrastructure
- Preprocessing pipeline must exactly match training data structure (no feature validation)
- No logging framework (uses print statements)
- Windows development environment (PowerShell paths in tests/workflows)
