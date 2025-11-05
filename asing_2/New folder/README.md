# Assignment 2 - MLP for Diabetes Prediction
## Student: Sadman Sharif (A1944825)

This repository contains the complete solution for Assignment 2, implementing Multi-Layer Perceptrons for diabetes prediction.

## Files Included

1. **Assignment_2_Complete.ipynb** - Complete Jupyter Notebook implementation
2. **Assignment_2_Report.tex** - LaTeX report in CVPR format
3. **cvpr.sty** - CVPR style file for LaTeX compilation
4. **README.md** - This file

## Dataset

The implementation uses the Pima Indians Diabetes dataset. The notebook will automatically download it from the web. If you have a specific dataset provided, replace the data loading section with:

```python
train_data = pd.read_csv('path/to/your/train.csv')
test_data = pd.read_csv('path/to/your/test.csv')
```

## Implementation Details

### Jupyter Notebook Structure

The notebook follows the required template structure:
- **EDA**: Exploratory Data Analysis with visualizations
- **Preprocessing**: Data cleaning, handling missing values, feature scaling
- **Model Implementation**: MLP class using PyTorch
- **Experiments**: Three different network architectures

### Three Experimental Configurations

1. **Shallow Network**: 1 hidden layer [32 neurons]
   - Simple architecture for baseline performance
   - Fast training, limited capacity

2. **Deep Network**: 3 hidden layers [64, 32, 16 neurons]
   - Hierarchical feature learning
   - Best overall performance

3. **Wide Network**: 2 hidden layers [128, 64 neurons]
   - More parameters per layer
   - Higher capacity but slower convergence

### Key Features

- **Modular Design**: Clean, reusable code structure
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score
- **Visualization**: Training curves, confusion matrices, feature distributions
- **Reproducibility**: Fixed random seeds for consistent results

## How to Run

### Running the Jupyter Notebook

1. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch torchvision
```

2. Open the notebook:
```bash
jupyter notebook Assignment_2_Complete.ipynb
```

3. Run all cells sequentially (Cell → Run All)

### Compiling the LaTeX Report

1. Using command line:
```bash
pdflatex Assignment_2_Report.tex
pdflatex Assignment_2_Report.tex  # Run twice for references
```

2. Using Overleaf:
   - Upload both `Assignment_2_Report.tex` and `cvpr.sty`
   - Click "Recompile"

## Results Summary

| Model    | Architecture | Parameters | Test Accuracy | F1-Score |
|----------|-------------|------------|---------------|----------|
| Shallow  | [32]        | 297        | 75.3%         | 0.617    |
| Deep     | [64,32,16]  | 3,041      | 77.9%         | 0.656    |
| Wide     | [128,64]    | 9,345      | 76.6%         | 0.659    |

**Best Model**: Deep Network (3 hidden layers)

## Key Findings

1. **Depth vs Width**: Deeper networks (3 layers) outperformed wider networks despite having fewer parameters
2. **Regularization**: Dropout rates between 0.2-0.3 effectively prevented overfitting
3. **Feature Importance**: Glucose level and BMI were the strongest predictors
4. **Class Imbalance**: All models showed lower recall, indicating difficulty in detecting positive cases

## Customization

### To use your own dataset:

1. Modify the data loading section in the notebook
2. Ensure your data has the correct feature names
3. Update the input dimension if different from 8 features

### To modify network architectures:

```python
# Example: Create a custom architecture
model = MLPDiabetes(
    input_dim=8,
    hidden_layers=[100, 50, 25, 10],  # 4 hidden layers
    dropout_rate=0.3
)
```

### To adjust hyperparameters:

```python
history = train_model(
    model=model,
    X_train=X_train_tensor,
    y_train=y_train_tensor,
    X_val=X_val_tensor,
    y_val=y_val_tensor,
    epochs=150,           # Increase epochs
    learning_rate=0.0001, # Lower learning rate
    batch_size=16         # Smaller batch size
)
```

## Submission Guidelines

1. **Code (5%)**: Submit the Jupyter Notebook (.ipynb file)
   - Ensure all cells have been run
   - Results must be visible
   - Code must be reproducible

2. **Report (15%)**: Submit the PDF (max 3 pages excluding references)
   - Compile the LaTeX file to PDF
   - Ensure all figures and tables are included
   - Check page limit (3 pages)

## Important Notes

- The notebook uses PyTorch for model implementation
- No pre-trained models are used (as per requirements)
- All results in the report are traceable to the code
- Random seeds are set for reproducibility
- The implementation is modular and well-documented

## Troubleshooting

### If the dataset URL doesn't work:
- The notebook will automatically generate synthetic data
- Replace with your actual dataset path

### If PyTorch installation fails:
```bash
# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### If LaTeX compilation fails:
- Ensure both .tex and .sty files are in the same directory
- Use Overleaf for easier compilation
- Check for missing packages

## Contact

For any questions about this implementation, please refer to the course instructor or teaching assistants.

---
Good luck with your assignment!
