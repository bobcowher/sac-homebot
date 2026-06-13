# test.py — watch the trained PointGoal reacher drive to random waypoints.
# Renders the env (human window) AND prints per-episode diagnostics so you can see
# WHERE the failures are: start distance, whether the straight line to the goal is
# blocked by a wall (line-of-sight), steps taken, reached/failed, final distance.
# Run download_models.sh first to pull the latest checkpoints.
from agent_reacher import ReacherAgent
from goal_geometry import GOAL_RADIUS, distance
import gymnasium as gym
import homebot  # noqa: F401


def make_env():
    for env_id in ("HomeBot2D-v1", "HomeBot2D-V1"):
        try:
            return gym.make(env_id, render_mode="human", action_mode="continuous",
                            obs_resolution=(96, 96), n_trash=2, max_steps=300,
                            map_name="default", goals=["trash"])
        except gym.error.Error:
            continue
    raise RuntimeError("No HomeBot2D env id registered")


def line_of_sight(base, gx, gy):
    """True if the straight line robot->goal does not cross a wall tile."""
    rx, ry = base._robot.x, base._robot.y
    solid = base._map.wall_solid
    ts = base._map.tile_size
    steps = max(2, int(distance(rx, ry, gx, gy) / (ts / 2)))
    for i in range(steps + 1):
        f = i / steps
        x, y = rx + (gx - rx) * f, ry + (gy - ry) * f
        col, row = int(x // ts), int(y // ts)
        if 0 <= row < solid.shape[0] and 0 <= col < solid.shape[1] and solid[row, col]:
            return False
    return True


env = make_env()
agent = ReacherAgent(env, max_buffer_size=1000)
agent.actor.load_the_model("goal_actor_best", device=agent.device)
agent.encoder.load_the_model("goal_encoder_best", device=agent.device)
agent.actor.eval(); agent.encoder.eval()

EPISODES = 20
reached_los = reached_blocked = n_los = n_blocked = 0
for ep in range(EPISODES):
    obs, _ = env.reset()
    base = env.unwrapped
    agent.goals.reset(base)
    obs = agent.process_observation(obs)
    gx, gy = agent.goals.goal_px
    start = distance(base._robot.x, base._robot.y, gx, gy)
    los = line_of_sight(base, gx, gy)
    reached, steps = False, 0
    for _ in range(300):
        img_t = (obs.unsqueeze(0).float() / 255.0).to(agent.device)
        action = agent._act(img_t, agent._goal_tensor(base), evaluate=True)
        nobs, _, _, trunc, _ = env.step(action)
        obs = agent.process_observation(nobs)
        steps += 1
        if distance(base._robot.x, base._robot.y, gx, gy) < GOAL_RADIUS:
            reached = True
            break
        if trunc:
            break
    final = distance(base._robot.x, base._robot.y, gx, gy)
    if los:
        n_los += 1; reached_los += reached
    else:
        n_blocked += 1; reached_blocked += reached
    tag = "LoS    " if los else "BLOCKED"
    print(f"ep {ep:2d} | {tag} | start {start:3.0f}px | "
          f"{'REACHED' if reached else 'failed '} in {steps:3d} steps | final {final:3.0f}px")

print(f"\nline-of-sight goals: {reached_los}/{n_los} reached")
print(f"blocked goals:       {reached_blocked}/{n_blocked} reached")
