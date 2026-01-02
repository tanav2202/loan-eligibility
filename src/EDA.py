import altair as alt
import pandas as pd
import numpy as np
import os
import click
from typing import List


class ExploratoryDataAnalysis():
    """Class containing static methods for generating EDA visualizations."""
    
    @staticmethod
    def univariate_feature_distributions(
            data: pd.DataFrame,
            column_names: List[str],
            bins: int = 30
        ) -> alt.Chart:
        """
        Plot univariate histograms for the given columns.
        
        Each feature gets its own histogram with its own x/y scale and a title.
        Charts are concatenated horizontally. This method is useful for understanding
        the distribution of individual numerical features.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame containing the columns to visualize.
        column_names : List[str]
            List of column names from the DataFrame to plot. All columns must exist
            in the DataFrame.
        bins : int, optional
            Maximum number of bins for each histogram. Default is 30.
        
        Returns
        -------
        alt.Chart
            A single Altair Chart object. Returns a single chart if only one column
            is provided, otherwise returns a horizontally concatenated chart.
        
        Raises
        ------
        ValueError
            If any column name is not found in the DataFrame.
        
        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({'Age': [25, 30, 35], 'Salary': [50000, 60000, 70000]})
        >>> chart = ExploratoryDataAnalysis.univariate_feature_distributions(
        ...     df, ['Age', 'Salary']
        ... )
        """
    
        for col in column_names:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
    
        charts = []
        for col in column_names:
            ch = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    x=alt.X(
                        f"{col}:Q",
                        bin=alt.Bin(maxbins=bins),
                        title="Value",
                        scale=alt.Scale(zero=False)  
                    ),
                    y=alt.Y(
                        "count():Q",
                        title="Count",
                        scale=alt.Scale(zero=False)
                    )
                )
                .properties(
                    title=col,
                    width=150,
                    height=120
                )
            )
            charts.append(ch)
    
        if len(charts) == 1:
            return charts[0]
        else:
            return alt.hconcat(*charts).properties(
                title="Univariate Feature Distributions"
            )
    
    @staticmethod
    def compare_categorical_features(
        data: pd.DataFrame,
        categorical_cols: List[str],
        target_name: str,
        columns: int = 3
    ) -> alt.VConcatChart:
        """
        Compare categorical features by plotting the mean numeric target value for each category.
        
        Produces a grid of bar charts showing how the target variable varies across
        different categories. Each categorical feature gets its own bar chart with
        category values on the x-axis and mean target value on the y-axis. Text labels
        display the exact mean values above each bar.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame containing both categorical features and the target variable.
        categorical_cols : List[str]
            List of categorical column names to analyze. All columns must exist in
            the DataFrame.
        target_name : str
            Name of the numeric target column. Must contain numeric values.
        columns : int, optional
            Number of charts per row in the grid layout. Default is 3.
        
        Returns
        -------
        alt.VConcatChart
            A vertically concatenated Altair chart object containing the grid of
            bar charts arranged in rows based on the columns parameter.
        
        Raises
        ------
        ValueError
            If target_name is not found in DataFrame.
            If target column is not numeric.
            If any categorical column is not found in DataFrame.
        
        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'Gender': ['M', 'F', 'M', 'F'],
        ...     'Married': ['Yes', 'No', 'Yes', 'No'],
        ...     'Loan_Status': [1.0, 0.0, 1.0, 0.0]
        ... })
        >>> chart = ExploratoryDataAnalysis.compare_categorical_features(
        ...     df, ['Gender', 'Married'], 'Loan_Status', columns=2
        ... )
        """

        if target_name not in data.columns:
            raise ValueError(f"Target column '{target_name}' not found in DataFrame.")

        if not np.issubdtype(data[target_name].dtype, np.number):
            raise ValueError("Target variable must be numeric for this plot.")

        for col in categorical_cols:
            if col not in data.columns:
                raise ValueError(f"Categorical column '{col}' not found in DataFrame.")

        charts = []

        for col in categorical_cols:
            # Compute mean target per category
            summary = (
                data
                .groupby(col)[target_name]
                .mean()
                .reset_index()
            )

            base = alt.Chart(summary).properties(width=180, height=140)

            bars = (
                base.mark_bar()
                .encode(
                    x=alt.X(f"{col}:N", title="Category"),
                    y=alt.Y(f"{target_name}:Q", title=f"Mean {target_name}"),
                    color=alt.Color(f"{target_name}:Q", scale=alt.Scale(scheme="blues"))
                )
            )

            labels = (
                base.mark_text(dy=-3, fontSize=11)
                .encode(
                    x=alt.X(f"{col}:N"),
                    y=alt.Y(f"{target_name}:Q"),
                    text=alt.Text(f"{target_name}:Q", format=".2f")
                )
            )

            chart = (bars + labels).properties(title=col)
            charts.append(chart)

        # Assemble into grid
        rows = []
        for i in range(0, len(charts), columns):
            row = alt.hconcat(*charts[i:i+columns])
            rows.append(row)

        final = (
            alt.vconcat(*rows)
            .resolve_scale()
            .configure_concat(spacing=20)
            .properties(
                title=f"Categorical Feature Comparison — Mean {target_name}"
            )
        )

        return final
    
    @staticmethod
    def density_feature_plots(
            data: pd.DataFrame,
            numerical_cols: List[str],
            target: str,
            columns: int = 3
        ) -> alt.VConcatChart:
        """
        Create density plots for numerical features, grouped by target classes.
        
        Generates kernel density estimation (KDE) plots for each numerical feature,
        with separate density curves for each class in the target variable. This helps
        visualize how feature distributions differ across target classes. Each subplot
        has independent axes.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame containing both numerical features and the target variable.
        numerical_cols : List[str]
            List of numerical column names to visualize. All columns must exist in
            the DataFrame and contain numeric values.
        target : str
            Name of the categorical target column for grouping density curves.
            Must exist in the DataFrame.
        columns : int, optional
            Number of density plots per row in the grid layout. Default is 3.
        
        Returns
        -------
        alt.VConcatChart
            A vertically concatenated Altair chart object containing the grid of
            density plots arranged in rows based on the columns parameter.
        
        Raises
        ------
        ValueError
            If target column is not found in DataFrame.
            If any numerical column is not found in DataFrame.
        
        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'Income': [50000, 60000, 70000, 55000],
        ...     'Loan_Status': [1, 0, 1, 0]
        ... })
        >>> chart = ExploratoryDataAnalysis.density_feature_plots(
        ...     df, ['Income'], 'Loan_Status'
        ... )
        """
    
        if target not in data.columns:
            raise ValueError(f"Target '{target}' not found in DataFrame.")
    
        for col in numerical_cols:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
    
        charts = []
    
        for col in numerical_cols:
            base = alt.Chart(data).properties(width=180, height=140)
    
            density = (
                base
                .transform_density(
                    col,
                    groupby=[target],
                    as_=["value", "density"]
                )
                .mark_line()
                .encode(
                    x=alt.X(
                        "value:Q",
                        title="Value",
                        scale=alt.Scale(zero=False)   
                    ),
                    y=alt.Y(
                        "density:Q",
                        title="Density",
                        scale=alt.Scale(zero=False)   
                    ),
                    color=alt.Color(f"{target}:N", title=target)
                )
                .properties(title=col)
            )
    
            charts.append(density)
    
        rows = []
        for i in range(0, len(charts), columns):
            rows.append(alt.hconcat(*charts[i:i + columns]))
    
        final = alt.vconcat(*rows).properties(
            title="Density Plots of Numerical Features Across Classes"
        )
    
        return final
    
    @staticmethod
    def boxplot_feature_plots(
            data: pd.DataFrame,
            numerical_cols: List[str],
            target: str,
            columns: int = 3
        ) -> alt.VConcatChart:
        """
        Create boxplots for numerical features, grouped by target classes.
        
        Generates boxplots for each numerical feature with separate boxes for each
        class in the target variable. Boxplots display the distribution, quartiles,
        and outliers for each feature-class combination. This is useful for identifying
        outliers and comparing distributions across classes.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame containing both numerical features and the target variable.
        numerical_cols : List[str]
            List of numerical column names to visualize. All columns must exist in
            the DataFrame and contain numeric values.
        target : str
            Name of the categorical target column for grouping boxplots.
            Must exist in the DataFrame.
        columns : int, optional
            Number of boxplots per row in the grid layout. Default is 3.
        
        Returns
        -------
        alt.VConcatChart
            A vertically concatenated Altair chart object containing the grid of
            boxplots arranged in rows based on the columns parameter.
        
        Raises
        ------
        ValueError
            If target column is not found in DataFrame.
            If any numerical column is not found in DataFrame.
        
        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'Age': [25, 30, 35, 40, 100],
        ...     'Approved': [1, 1, 0, 1, 0]
        ... })
        >>> chart = ExploratoryDataAnalysis.boxplot_feature_plots(
        ...     df, ['Age'], 'Approved'
        ... )
        """
    
        if target not in data.columns:
            raise ValueError(f"Target '{target}' not found in DataFrame.")
    
        for col in numerical_cols:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
    
        charts = []
    
        for col in numerical_cols:
            base = alt.Chart(data).properties(width=180, height=140)
    
            box = (
                base.mark_boxplot()
                .encode(
                    x=alt.X(
                        f"{target}:N",
                        title=target
                    ),
                    y=alt.Y(
                        f"{col}:Q",
                        title="Value",
                        scale=alt.Scale(zero=False)     
                    ),
                    color=alt.Color(
                        f"{target}:N",
                        title=target
                    )
                )
                .properties(title=col)
            )
    
            charts.append(box)
    
        rows = []
        for i in range(0, len(charts), columns):
            rows.append(alt.hconcat(*charts[i:i + columns]))
    
        final = alt.vconcat(*rows).properties(
            title="Boxplots of Numerical Features Across Classes"
        )
    
        return final
    
    @staticmethod
    def correlation_plot(
            data: pd.DataFrame, 
            column_names: List[str]
        ) -> alt.Chart:
        """
        Create a correlation heatmap for selected numerical features.
        
        Generates a color-coded heatmap showing pairwise Pearson correlation
        coefficients between numerical features. Correlations range from -1 (perfect
        negative) to +1 (perfect positive). Red indicates positive correlations,
        blue indicates negative correlations, and white indicates zero correlation.
        Text labels display exact correlation values.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame containing the numerical features to correlate.
        column_names : List[str]
            List of numerical column names to include in correlation analysis.
            All columns must exist in the DataFrame and contain numeric values.
        
        Returns
        -------
        alt.Chart
            An Altair chart object displaying the correlation heatmap with
            colored rectangles and text annotations.
        
        Raises
        ------
        ValueError
            If any column name is not found in the DataFrame.
        
        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'Age': [25, 30, 35],
        ...     'Income': [50000, 60000, 70000],
        ...     'Years_Experience': [2, 5, 8]
        ... })
        >>> chart = ExploratoryDataAnalysis.correlation_plot(
        ...     df, ['Age', 'Income', 'Years_Experience']
        ... )
        """
    
        for col in column_names:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
    
        corr = data[column_names].corr()
    
        corr_long = (
            corr
            .reset_index()
            .melt(id_vars="index", var_name="feature2", value_name="correlation")
            .rename(columns={"index": "feature1"})
        )
    
        heatmap = (
            alt.Chart(corr_long)
            .mark_rect()
            .encode(
                x=alt.X("feature1:N", title=None, sort=column_names),
                y=alt.Y("feature2:N", title=None, sort=column_names),
                color=alt.Color(
                    "correlation:Q",
                    scale=alt.Scale(scheme="redblue", domain=(-1, 1)),
                    title="Correlation"
                ),
            )
            .properties(width=300, height=300)
        )
    
        text = (
            alt.Chart(corr_long)
            .mark_text(baseline="middle")
            .encode(
                x=alt.X("feature1:N", sort=column_names),
                y=alt.Y("feature2:N", sort=column_names),
                text=alt.Text("correlation:Q", format=".2f"),
                color=alt.condition(
                    "datum.correlation > 0",
                    alt.value("black"),
                    alt.value("white")
                )
            )
        )
    
        final = (
            (heatmap + text)
            .properties(
                title="Correlation Heatmap"
            )
        )
    
        return final


@click.command()
@click.option(
    '--train-data',
    type=str,
    required=True,
    help="Path to processed training data CSV (e.g., df_train.csv)"
)
@click.option(
    '--output-dir',
    type=str,
    default="results/figures",
    help="Directory where PNG plots will be saved"
)
def main(train_data, output_dir):
    """
    Generate exploratory data analysis (EDA) visualizations.
    
    This script creates the following plots:
    1. Univariate distributions of numerical features
    2. Categorical feature comparison by loan status
    3. Density plots by loan status
    4. Boxplots showing outliers
    5. Correlation heatmap
    
    All plots are saved as PNG files in the specified output directory.
    
    Examples:
        python scripts/EDA.py \\
            --train-data data/processed/df_train.csv \\
            --output-dir results/figures
    """

    # Load data
    print(f"Loading training data from: {train_data}")
    if not os.path.exists(train_data):
        raise FileNotFoundError(f"Training data file not found: {train_data}")
    
    data = pd.read_csv(train_data)
    print(f"  Shape: {data.shape}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Define feature groups
    numerical_cols = [
        'Applicant_Income', 
        'Coapplicant_Income', 
        'Loan_Amount', 
        'Loan_Amount_Term', 
        'Dependents'
    ]
    
    categorical_cols = [
        'Credit_History', 
        'Gender', 
        'Married', 
        'Education', 
        'Self_Employed',
        'Property_Area'
    ]
    
    target_name = 'Loan_Status'

    print("\nGenerating EDA plots...")

    # Build and save plots
    charts = {
        "univariate.png": ExploratoryDataAnalysis.univariate_feature_distributions(
            data, numerical_cols
        ),
        "categorical_compare.png": ExploratoryDataAnalysis.compare_categorical_features(
            data, categorical_cols, target_name
        ),
        "density_plots.png": ExploratoryDataAnalysis.density_feature_plots(
            data, numerical_cols, target_name
        ),
        "boxplots.png": ExploratoryDataAnalysis.boxplot_feature_plots(
            data, numerical_cols, target_name
        ),
        "correlation_heatmap.png": ExploratoryDataAnalysis.correlation_plot(
            data, numerical_cols
        ),
    }

    # Save charts
    for filename, chart in charts.items():
        out_path = os.path.join(output_dir, filename)
        chart.save(out_path)
        print(f"   Saved: {out_path}")

    print(f"\n All EDA plots generated successfully in: {output_dir}")


if __name__ == "__main__":
    main()