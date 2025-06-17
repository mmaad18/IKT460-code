import plotly.express as px
import pandas as pd

import numpy as np
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def plot_episode_metrics(episode_data: dict[int, list[dict]]) -> None:
    episode_metrics = []

    for episode_num, step_infos in episode_data.items():
        if not step_infos:
            continue

        step_count = step_infos[-1]['step_count']
        total_reward = sum(step['reward'] for step in step_infos)
        total_time_penalty = sum(step['reward_components'][0] for step in step_infos)
        total_omega_penalty = sum(step['reward_components'][1] for step in step_infos)
        total_collision_penalty = sum(step['reward_components'][2] for step in step_infos)
        total_velocity_reward = sum(step['reward_components'][3] for step in step_infos)
        total_coverage_reward = sum(step['reward_components'][4] for step in step_infos)
        final_coverage = step_infos[-1]['coverage']

        episode_metrics.append({
            'episode': episode_num,
            'total_reward': total_reward,
            'total_time_penalty': total_time_penalty,
            'total_omega_penalty': total_omega_penalty,
            'total_collision_penalty': total_collision_penalty,
            'total_velocity_reward': total_velocity_reward,
            'total_coverage_reward': total_coverage_reward,
            'final_coverage': final_coverage,
            'step_count': step_count
        })

    metrics_df = pd.DataFrame(episode_metrics)

    fig = make_subplots(
        rows=8, cols=1,
        subplot_titles=('Total Reward per Episode', 'Final Coverage per Episode', 'Step Count per Episode', 'Total Time Penalty per Episode',
                        'Total Omega Penalty per Episode', 'Total Collision Penalty per Episode', 'Total Velocity Reward per Episode', 'Total Coverage Reward per Episode'),
        vertical_spacing=0.04
    )

    fig.add_trace(
        go.Scatter(x=metrics_df['episode'], y=metrics_df['total_reward'],
                   mode='markers', name='Total Reward', line=dict(color='blue')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=metrics_df['episode'], y=metrics_df['final_coverage'],
                   mode='markers', name='Final Coverage', line=dict(color='green')),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=metrics_df['episode'], y=metrics_df['step_count'],
                   mode='markers', name='Step Count', line=dict(color='red')),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(x=metrics_df['episode'], y=metrics_df['total_time_penalty'],
                   mode='markers', name='Total Time Penalty', line=dict(color='orange')),
        row=4, col=1
    )

    fig.add_trace(
        go.Scatter(x=metrics_df['episode'], y=metrics_df['total_omega_penalty'],
                   mode='markers', name='Total Angular Velocity Penalty', line=dict(color='purple')),
        row=5, col=1
    )

    fig.add_trace(
        go.Scatter(x=metrics_df['episode'], y=metrics_df['total_collision_penalty'],
                   mode='markers', name='Total Collision Penalty', line=dict(color='black')),
        row=6, col=1
    )

    fig.add_trace(
        go.Scatter(x=metrics_df['episode'], y=metrics_df['total_velocity_reward'],
                   mode='markers', name='Total Velocity Reward', line=dict(color='cyan')),
        row=7, col=1
    )

    fig.add_trace(
        go.Scatter(x=metrics_df['episode'], y=metrics_df['total_coverage_reward'],
                   mode='markers', name='Total Coverage Reward', line=dict(color='magenta')),
        row=8, col=1
    )

    fig.update_layout(
        height=1600,
        title_text="Episode Performance Metrics",
        showlegend=False
    )

    # Update x-axis labels
    fig.update_xaxes(title_text="Episode Number", row=8, col=1)

    # Update y-axis labels
    fig.update_yaxes(title_text="Total Reward", row=1, col=1)
    fig.update_yaxes(title_text="Coverage", row=2, col=1)
    fig.update_yaxes(title_text="Steps", row=3, col=1)
    fig.update_yaxes(title_text="Time Penalty", row=4, col=1)
    fig.update_yaxes(title_text="Omega Penalty", row=5, col=1)
    fig.update_yaxes(title_text="Collision Penalty", row=6, col=1)
    fig.update_yaxes(title_text="Velocity Reward", row=7, col=1)
    fig.update_yaxes(title_text="Coverage Reward", row=8, col=1)

    fig.show()


def plot_reward_components(episode_data: dict[int, list[dict]], episode_idx: int) -> None:
    df = episode_data_to_dataframe(episode_data)
    episode_df = df[df["episode"] == episode_idx].copy()
    episode_df[["time", "omega", "collision", "velocity", "coverage"]] = episode_df["reward_components"].apply(pd.Series)
    map_name = episode_df["environment_name"].iloc[0]

    fig = make_subplots(
        rows=5, cols=1,
        subplot_titles=('Time Penalty per Step', 'Omega Penalty per Step', 'Collision Penalty per Step', 'Velocity Reward per Step', 'Coverage Reward per Step', 'Q-Values per Step'),
        vertical_spacing=0.04
    )

    fig.add_trace(
        go.Scatter(x=episode_df['step_count'], y=episode_df['time'],
                   mode='markers', name='Time Penalty', line=dict(color='blue')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=episode_df['step_count'],  y=episode_df['omega'],
                   mode='markers', name='Omega Penalty', line=dict(color='orange')),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=episode_df['step_count'],  y=episode_df['collision'],
                   mode='markers', name='Collision Penalty', line=dict(color='red')),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(x=episode_df['step_count'],  y=episode_df['velocity'],
                   mode='markers', name='Velocity Reward', line=dict(color='green')),
        row=4, col=1
    )

    fig.add_trace(
        go.Scatter(x=episode_df['step_count'],  y=episode_df['coverage'],
                   mode='markers', name='Coverage Reward', line=dict(color='purple')),
        row=5, col=1
    )

    fig.update_layout(
        height=1000,
        title_text=f"Reward Components per Step, Map: {map_name}, Episode {episode_idx}",
        showlegend=False
    )

    # Update x-axis labels
    fig.update_xaxes(title_text="Step Count", row=6, col=1)

    # Update y-axis labels
    fig.update_yaxes(title_text="Time Penalty", row=1, col=1)
    fig.update_yaxes(title_text="Omega Penalty", row=2, col=1)
    fig.update_yaxes(title_text="Collision Penalty", row=3, col=1)
    fig.update_yaxes(title_text="Velocity Reward", row=4, col=1)
    fig.update_yaxes(title_text="Coverage Reward", row=5, col=1)

    fig.show()


def plot_velocity(episode_data: dict[int, list[dict]], episode_idx: int) -> None:
    df = episode_data_to_dataframe(episode_data)
    episode_df = df[df["episode"] == episode_idx].copy()
    episode_df[["velocity_x", "velocity_y", "omega"]] = episode_df["agent_local_velocity"].apply(pd.Series)
    map_name = episode_df["environment_name"].iloc[0]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Velocity X per Step', 'Omega per Step'),
        vertical_spacing=0.06
    )

    fig.add_trace(
        go.Scatter(x=episode_df['step_count'], y=episode_df['velocity_x'],
                   mode='markers', name='Velocity X', line=dict(color='blue')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=episode_df['step_count'], y=episode_df['omega'],
                   mode='markers', name='Angular Velocity (Omega)', line=dict(color='red')),
        row=2, col=1
    )

    fig.update_layout(
        height=800,
        title_text=f"Agent Velocities per Step, Map: {map_name}, Episode {episode_idx}",
        showlegend=False
    )

    # Update x-axis labels
    fig.update_xaxes(title_text="Step Count", row=3, col=1)

    # Update y-axis labels
    fig.update_yaxes(title_text="Velocity X", row=1, col=1)
    fig.update_yaxes(title_text="Omega", row=3, col=1)

    fig.show()


def plot_pose(episode_data: dict[int, list[dict]], episode_idx: int) -> None:
    df  = episode_data_to_dataframe(episode_data)
    episode_df = df[df["episode"] == episode_idx].copy()
    episode_df[["x", "y", "th"]] = episode_df["agent_pose"].apply(pd.Series)
    map_name = episode_df["environment_name"].iloc[0]

    fig = px.scatter(episode_df, x="x", y="y", color="step_count", color_continuous_scale=px.colors.sequential.Viridis)

    fig.update_layout(
        width=1200,
        height=600,
        title_text=f"Agent Pose per Step, Map: {map_name}, Episode {episode_idx}",
        showlegend=False,
        xaxis=dict(title="X Position", range=[0, 1200]),
        yaxis=dict(title="Y Position", range=[0, 600], scaleanchor="x", scaleratio=1)
    )

    fig.update_yaxes(autorange="reversed")

    fig.show()


def plot_q_values(episode_data: dict[int, list[dict]], episode_idx: int) -> None:
    df = episode_data_to_dataframe(episode_data)
    episode_df = df[df["episode"] == episode_idx].copy()

    episode_df = episode_df[episode_df["q_values"].apply(lambda q: isinstance(q, list) and len(q) > 0)]
    episode_df["max_q"] = episode_df["q_values"].apply(max)
    episode_df["min_q"] = episode_df["q_values"].apply(min)
    episode_df["mean_q"] = episode_df["q_values"].apply(lambda q: sum(q) / len(q))

    map_name = episode_df["environment_name"].iloc[0]

    fig = px.line(
        episode_df,
        x="step_count",
        y=["min_q", "mean_q", "max_q"],
        labels={"value": "Q-Value", "variable": "Q-Stat"},
        title=f"Q-Values per Step, Map: {map_name}, Episode {episode_idx}",
    )

    fig.update_layout(height=600)
    
    fig.show()


def main() -> None:
    run_id = "run_250617_141429"    
    episode_data = load_all_episode_data(f"project/output/logs/{run_id}")
    df = episode_data_to_dataframe(episode_data)
    
    if False:
        plot_episode_metrics(episode_data)
    else:
        plot_reward_components(episode_data, 8900)
        plot_velocity(episode_data, 8900)
        plot_pose(episode_data, 8900)
        plot_q_values(episode_data, 8900)


main()

