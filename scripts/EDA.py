

import altair as alt
import pandas as pd
from typing import List

class EDA():
    
    @staticmethod
    def univariate_feature_distributions(
            data: pd.DataFrame,
            column_names: List[str],
            bins: int = 30
        ) -> alt.Chart:
        """
        Plot univariate histograms for the given columns.
        Each feature gets its own histogram with its own x/y scale and a title.
        Charts are concatenated horizontally.
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
        Compare binary/categorical features graphically across classes in `target_name`.
        Produces a grid of bar charts (one per feature), each with independent scales.
        """
    
        if target_name not in data.columns:
            raise ValueError(f"target_name '{target_name}' not found.")
    
        for col in categorical_cols:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found.")
    
        charts = []
        for col in categorical_cols:
    
            base = alt.Chart(data).properties(width=180, height=140)
    
            bars = (
                base.mark_bar()
                .encode(
                    x=alt.X(f"{col}:N", title="Category"),
                    y=alt.Y("count():Q", title="Count"),
                    color=alt.Color(f"{target_name}:N", title=target_name)
                )
            )
    
            labels = (
                base.mark_text(dy=-3, fontSize=11)
                .encode(
                    x=alt.X(f"{col}:N"),
                    y=alt.Y("count():Q"),
                    text=alt.Text("count():Q")
                )
            )
    
            chart = (bars + labels).properties(title=col)
            charts.append(chart)
    
        rows = []
        for i in range(0, len(charts), columns):
            row = alt.hconcat(*charts[i:i+columns])
            rows.append(row)
    
        final = alt.vconcat(*rows).properties(
            title="Categorical Feature Comparison Across Target Classes"
        )
    
        return final
    
    @staticmethod
    def density_feature_plots(
            data: pd.DataFrame,
            numerical_cols: List[str],
            target: str,
            columns: int = 3
        ):
        """
        Create density plots for each numeric feature in `numerical_cols`,
        grouped by classes in `target`. Independent axes per subplot.
        Manual grid layout (no facet).
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
        ):
        """
        Create boxplots for each numeric feature in `numerical_cols`,
        grouped by classes in `target`. Independent axes per subplot.
        Manual grid layout (no facet).
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
        Create a correlation heatmap for the selected feature names.
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


