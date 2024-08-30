from gymnasium.envs.registration import register
from .lqr import LQR

register(
    id='LQR-v0',
    entry_point='potion.envs.lqr:LQR'
)

register(
    id='CartPoleContinuous-v0',
    entry_point='potion.envs.cartpole_continuous:CartPoleContinuous'
)

register(
    id='GridWorld-v0',
    entry_point='potion.envs.gridworld:GridWorld'
)

register(
    id='Minigolf-v0',
    entry_point='potion.envs.minigolf:Minigolf'
)
