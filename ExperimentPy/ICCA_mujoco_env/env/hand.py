import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box


class HandEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(self,**kwargs):
        utils.EzPickle.__init__(self)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(48,), dtype=np.float64)
        mujoco_env.MujocoEnv.__init__(self, "Hand.xml", 2,observation_space=observation_space,**kwargs)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        reward = 0
        return ob, reward, False, done,{}

    def viewer_setup(self): #to do
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):#to do
        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos
        )
        while True:
            self.goal = self.np_random.uniform(low=-0.2, high=0.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qvel = self.init_qvel

        self.set_state(np.array(qpos), np.array(qvel))
        ob =  self._get_obs()
        return ob

    def _get_obs(self):
        return self.data.qpos.flat.copy().ravel()
