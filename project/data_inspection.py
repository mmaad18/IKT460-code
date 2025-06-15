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

    # Create subplots
    fig = make_subplots(
        rows=8, cols=1,
        subplot_titles=('Total Reward per Episode', 'Final Coverage per Episode', 'Step Count per Episode', 'Total Time Penalty per Episode',
                        'Total Omega Penalty per Episode', 'Total Collision Penalty per Episode', 'Total Velocity Reward per Episode', 'Total Coverage Reward per Episode'),
        vertical_spacing=0.04
    )

    # Plot total reward
    fig.add_trace(
        go.Scatter(x=metrics_df['episode'], y=metrics_df['total_reward'],
                   mode='markers', name='Total Reward', line=dict(color='blue')),
        row=1, col=1
    )

    # Plot final coverage
    fig.add_trace(
        go.Scatter(x=metrics_df['episode'], y=metrics_df['final_coverage'],
                   mode='markers', name='Final Coverage', line=dict(color='green')),
        row=2, col=1
    )

    # Plot step count
    fig.add_trace(
        go.Scatter(x=metrics_df['episode'], y=metrics_df['step_count'],
                   mode='markers', name='Step Count', line=dict(color='red')),
        row=3, col=1
    )

    # Plot total time penalty
    fig.add_trace(
        go.Scatter(x=metrics_df['episode'], y=metrics_df['total_time_penalty'],
                   mode='markers', name='Total Time Penalty', line=dict(color='orange')),
        row=4, col=1
    )

    # Plot total omega penalty
    fig.add_trace(
        go.Scatter(x=metrics_df['episode'], y=metrics_df['total_omega_penalty'],
                   mode='markers', name='Total Angular Velocity Penalty', line=dict(color='purple')),
        row=5, col=1
    )

    # Plot total collision penalty
    fig.add_trace(
        go.Scatter(x=metrics_df['episode'], y=metrics_df['total_collision_penalty'],
                   mode='markers', name='Total Collision Penalty', line=dict(color='black')),
        row=6, col=1
    )

    # Plot total velocity reward
    fig.add_trace(
        go.Scatter(x=metrics_df['episode'], y=metrics_df['total_velocity_reward'],
                   mode='markers', name='Total Velocity Reward', line=dict(color='cyan')),
        row=7, col=1
    )

    # Plot total coverage reward
    fig.add_trace(
        go.Scatter(x=metrics_df['episode'], y=metrics_df['total_coverage_reward'],
                   mode='markers', name='Total Coverage Reward', line=dict(color='magenta')),
        row=8, col=1
    )

    # Update layout
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

    fig = make_subplots(
        rows=5, cols=1,
        subplot_titles=('Time Penalty per Step', 'Omega Penalty per Step', 'Collision Penalty per Step', 'Velocity Reward per Step', 'Coverage Reward per Step'),
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

    # Update layout
    fig.update_layout(
        height=1000,
        title_text=f"Reward Components per Step, Episode {episode_idx}",
        showlegend=False
    )

    # Update x-axis labels
    fig.update_xaxes(title_text="Step Count", row=5, col=1)

    # Update y-axis labels
    fig.update_yaxes(title_text="Time Penalty", row=1, col=1)
    fig.update_yaxes(title_text="Omega Penalty", row=2, col=1)
    fig.update_yaxes(title_text="Collision Penalty", row=3, col=1)
    fig.update_yaxes(title_text="Velocity Reward", row=4, col=1)
    fig.update_yaxes(title_text="Coverage Reward", row=5, col=1)

    fig.show()


def main() -> None:
    episode_data = load_all_episode_data("project/output/logs/run_d8a6bd1c-9137-4ef6-a7d5-af2285a88800")
    df = episode_data_to_dataframe(episode_data)
    
    # Reward 
    #fig = px.line(df, x="step_count", y="reward", color="episode", title="Reward per Step")
    #fig.update_layout(legend=dict(title="Episode", itemsizing='constant'))
    
    # Coverage
    #fig = px.line(df, x="step_count", y="coverage", color="episode", title="Coverage per Step")
    #fig.update_layout(legend=dict(title="Episode", itemsizing='constant'))
    
    # Duration
    #fig = px.line(df, x="step_count", y="elapsed_time", color="episode", title="Elapsed Time per Step")
    #fig.update_layout(legend=dict(title="Episode", itemsizing='constant'))
    
    # Velocity
    #df[["velocity_x", "velocity_y", "omega"]] = df["agent_local_velocity"].apply(pd.Series)
    #fig = px.line(df, x="step_count", y="velocity_x", color="episode", title="Agent Velocity per Step")
    #fig = px.line(df, x="step_count", y="omega", color="episode", title="Angular Velocity (Omega) per Step")
    
    #fig.show()
    
    #plot_episode_metrics(episode_data)
    plot_reward_components(episode_data, 1800)


main()

