import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import pandas as pd

from phase8_enhancements.active_relabel import ActiveRelabelQueue
from phase8_enhancements.auto_cleaning import AutoCleaningAdvisor


class LQNEWidget:
    def __init__(self, df):
        self.df = df

        self.trust_slider = widgets.FloatSlider(
            value=0.25, min=0.05, max=0.5, step=0.05,
            description="Trust Threshold", continuous_update=False
        )

        self.drop_slider = widgets.FloatSlider(
            value=0.01, min=0.0, max=0.05, step=0.005,
            description="Drop Fraction", continuous_update=False
        )

        self.topk_slider = widgets.IntSlider(
            value=200, min=50, max=1000, step=50,
            description="Top-K Relabel", continuous_update=False
        )

        self.run_button = widgets.Button(
            description="Run LQNE",
            button_style="success",
        )

        self.output = widgets.Output()
        self.run_button.on_click(self._run)

    def display(self):
        display(widgets.VBox([
            self.trust_slider,
            self.drop_slider,
            self.topk_slider,
            self.run_button,
            self.output
        ]))

    def _run(self, _):
        with self.output:
            clear_output()
            print("🔍 Running LQNE analysis...\n")

            # Relabeling
            relabeler = ActiveRelabelQueue()
            relabel_df = relabeler.generate(
                self.df,
                top_k=self.topk_slider.value
            )

            # Cleaning
            cleaner = AutoCleaningAdvisor(
                trust_threshold=self.trust_slider.value,
                drop_fraction=self.drop_slider.value,
            )
            cleaning = cleaner.generate(self.df)

            # ---- Charts ----
            plt.figure()
            self.df["trust_score"].hist(bins=30)
            plt.title("Trust Score Distribution")
            plt.xlabel("Trust Score")
            plt.ylabel("Count")
            plt.show()

            # ---- Tables ----
            print("\n📌 Top Relabel Candidates")
            display(relabel_df.head(10))

            print("\n📊 Dataset Health Summary")
            display(pd.DataFrame([cleaning["summary"]]))
