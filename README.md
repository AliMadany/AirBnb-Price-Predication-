
> **Note:** Replace the filenames above with your actual CSV filenames. The notebook’s code references `folder_path = '/content/Dataset'`, so if you move the notebook off of Google Colab, update `folder_path` to the local path (e.g., `./Dataset`) accordingly.

Typical columns in the raw listings CSVs include (but are not limited to):
- `id`
- `name`
- `latitude`, `longitude`
- `room_type`
- `person_capacity`
- `price` (or a similar target feature like `realSum`)
- `guest_satisfaction_overall`
- `number_of_reviews`
- `availability_365`
- ...and more.

---

## Notebook Structure

Below is a high‐level outline of the sections in **`ML.ipynb`**:

1. **Import Libraries**  
   - `pandas`, `numpy`, `os` for data handling  
   - `seaborn`, `matplotlib.pyplot` for visualization  
   - `sklearn` modules for preprocessing, dimensionality reduction, clustering, and modeling

2. **Import Datasets**  
   - Load all CSV files from `Dataset/` into individual DataFrames  
   - Store them in a list, then concatenate into a single “master” DataFrame

3. **Combine Datasets**  
   - Merge or concatenate as needed (e.g., join listings with reviews, if applicable)  
   - Drop duplicate rows (if any)

4. **Data Exploration & Visualization**  
   - Display head/tail of data, summary statistics (`.describe()`), data types  
   - Visualize distributions of numerical features (histograms, boxplots)  
   - Plot categorical counts or bar charts for room types, neighborhoods, etc.

5. **Data Cleaning**  
   - Check for null/missing values and impute or drop as appropriate  
   - Detect outliers (IQR method, z-score) and decide whether to cap, remove, or keep them  
   - Drop irrelevant columns (e.g., IDs or columns with too many missing values)

6. **Feature Selection / Engineering**  
   - Correlation matrix to identify highly correlated features  
   - Possibly drop or combine features for multicollinearity reduction  
   - Create new features (e.g., total stay cost, review rates, etc.)

7. **One-Hot Encoding**  
   - Convert categorical variables (e.g., `room_type`, `neighborhood`, `cancellation_policy`) into dummy/indicator variables

8. **Feature Scaling**  
   - Standardize numerical features using `StandardScaler`  
   - (Optional) Compare StandardScaler vs. MinMaxScaler if necessary

9. **Data Splitting**  
   - Define `X` (feature matrix) and `y` (target vector, e.g., `price` or `realSum`)  
   - Split into training and testing sets via `train_test_split(test_size=0.20, random_state=42)`

10. **PCA Analysis**  
    - Fit `PCA` on the scaled training features  
    - Examine explained variance ratio, generate a scree plot  
    - Transform `X_train` and `X_test` into the reduced principal component space

11. **Clustering**  
    - Apply `KMeans(n_clusters=k, random_state=42)` on PCA‐transformed features  
    - Compute silhouette score to choose an optimal `k`  
    - Produce a cluster summary table (mean values of important features per cluster)  
    - Visualize clusters (e.g., colors on two leading principal components)

12. **Modeling (Regression)**  
    - Instantiate multiple regression models:  
      - `LinearRegression()`  
      - `Ridge(alpha=1.0)`  
      - `Lasso(alpha=1.0)`  
    - Train each on `X_train, y_train`  
    - Compute metrics on both train/test sets:  
      - Mean Squared Error (MSE)  
      - Mean Absolute Error (MAE)  
      - R² score  
    - Tabulate results in a Pandas DataFrame and print comparisons

13. **Conclusions & Next Steps**  
    - Interpret which model performed best  
    - Outline suggestions for future work: hyperparameter tuning, additional features, different clustering algorithms, or even deep learning approaches

---

## Installation & Setup

1. **Clone this repository** (or download the `.ipynb` and dataset folder):
   ```bash
   git clone https://github.com/YourUsername/airbnb-ml-project.git
   cd airbnb-ml-project
