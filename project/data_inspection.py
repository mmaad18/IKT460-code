import plotly.express as px
import pandas as pd

import numpy as np
from pathlib import Path


def load_all_episode_data(run_folder: str) -> dict[int, list[dict]]:
    run_path = Path(run_folder)
    episode_data: dict[int, list[dict]] = {}

    for file in sorted(run_path.glob("episode_*_data.npz")):
        episode_num = int(file.stem.split("_")[1])
        loaded = np.load(file, allow_pickle=True)
        step_infos = list(loaded["data"])
        episode_data[episode_num] = step_infos

    return episode_data


def episode_data_to_dataframe(episode_data: dict[int, list[dict]]) -> pd.DataFrame:
    all_step_infos = []
    for episode_num, step_infos in episode_data.items():
        for step_info in step_infos:
            step_info_with_episode = step_info.copy()
            step_info_with_episode['episode'] = episode_num
            all_step_infos.append(step_info_with_episode)

    return pd.DataFrame(all_step_infos)


def main():
    episode_data = load_all_episode_data("project/output/logs/run_5c9343a8-6989-4007-932d-c1b644c8acfd")
    df = episode_data_to_dataframe(episode_data)
    
    # Reward 
    #fig = px.line(df, x="step_count", y="reward", color="episode", title="Reward per Step")
    #fig.update_layout(legend=dict(title="Episode", itemsizing='constant'))
    
    # Coverage
    fig = px.line(df, x="step_count", y="coverage", color="episode", title="Coverage per Step")
    fig.update_layout(legend=dict(title="Episode", itemsizing='constant'))
    
    # Duration
    #fig = px.line(df, x="step_count", y="elapsed_time", color="episode", title="Elapsed Time per Step")
    #fig.update_layout(legend=dict(title="Episode", itemsizing='constant'))
    
    fig.show()


main()

