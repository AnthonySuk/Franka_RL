# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .base.legged_robot import LeggedRobot
from .tita.tita import Tita
from .tita.tita_rough_config import TitaRoughCfg, TitaRoughCfgPPO
from .tita.tita_flat_config import TitaFlatCfg, TitaFlatCfgPPO
from .airbot.tita_airbot.tita_airbot_rough_only_config import TitaAirbotRoughOnlyCfg, TitaAirbotRoughOnlyCfgPPO

from legged_gym.utils.task_registry import task_registry
task_registry.register("tita_flat", Tita, TitaFlatCfg(), TitaFlatCfgPPO())
task_registry.register("tita_rough", Tita, TitaRoughCfg(), TitaRoughCfgPPO())

from .tita_airbot.tita_airbot_robot import TitaAirbotRobot
from .tita_airbot.tita_airbot_rough_config import TitaAirbotRoughCfg, TitaAirbotRoughCfgPPO
from .tita_airbot.tita_airbot_flat_config import TitaAirbotFlatCfg, TitaAirbotFlatCfgPPO

task_registry.register("tita_airbot_flat", TitaAirbotRobot, TitaAirbotFlatCfg(), TitaAirbotFlatCfgPPO())
task_registry.register("tita_airbot_rough", TitaAirbotRobot, TitaAirbotRoughCfg(), TitaAirbotRoughCfgPPO())

from .tita_airbot_copy.tita_airbot_follow_robot import TitaAirbotRobot_F
from .tita_airbot_copy.tita_airbot_follow_config import TitaAirbotFollowCfg, TitaAirbotFollowCfgPPO

task_registry.register("tita_airbot_follow", TitaAirbotRobot_F, TitaAirbotFollowCfg(), TitaAirbotFollowCfgPPO())

from .tita_airbot_copy.tita_airbot_follow_constrain_robot import TitaAirbotRobotConstrain_F
from .tita_airbot_copy.tita_airbot_follow_constrain_config import TitaAirbotFollowConstrainCfg, TitaAirbotFollowConstrainCfgPPO
task_registry.register("tita_airbot_constrain", TitaAirbotRobotConstrain_F, TitaAirbotFollowConstrainCfg(), TitaAirbotFollowConstrainCfgPPO())

from .tita_airbot_copy.TAF_visual_wholebody_robot import TAFVisualWholebodyRobot
from .tita_airbot_copy.TAF_visual_wholebody_config import TAFVisualWholebodyCfg, TAFVisualWholebodyCfgPPO
task_registry.register("tita_visual_wholebody", TAFVisualWholebodyRobot, TAFVisualWholebodyCfg(), TAFVisualWholebodyCfgPPO())

from .airbot.tita_airbot.tita_airbot_only_robot import TitaAirbotOnlyRobot
from .airbot.tita_airbot.tita_airbot_only_config import TitaAirbotOnlyCfg, TitaAirbotOnlyCfgPPO
task_registry.register("tita_airbot_only",TitaAirbotOnlyRobot,TitaAirbotOnlyCfg(),TitaAirbotOnlyCfgPPO())

from .tita_airbot_np3o.tita_airbot_np3o_config import TitaAirbotNP3OCfg, TitaAirbotNP3OCfgPPO
from .tita_airbot_np3o.tita_airbot_robot_np3o import TitaAirbotRobotNP3O
task_registry.register("tita_airbot_np3o", TitaAirbotRobotNP3O, TitaAirbotNP3OCfg(), TitaAirbotNP3OCfgPPO())

from .franka.franka_only_config import FrankaOnlyCfg, FrankaOnlyCfgPPO
from .franka.franka_only_robot import FrankaOnlyRobot
task_registry.register("franka_only", FrankaOnlyRobot, FrankaOnlyCfg(), FrankaOnlyCfgPPO())

# import os
# import sys
# robot_type = os.getenv("ROBOT_TYPE")
# print(robot_type, "in env __init__")
# Check if the ROBOT_TYPE environment variable is set, otherwise exit with an error
# if not robot_type:
    # print("Error: Please set the ROBOT_TYPE using 'export ROBOT_TYPE=<robot_type>'.")
# else:
#     if robot_type.startswith("PF"):
#         from .pointfoot.PF.pointfoot import PointFoot
#         from legged_gym.envs.pointfoot.mixed_terrain.pointfoot_rough_config import PointFootRoughCfg, PointFootRoughCfgPPO
#         from legged_gym.envs.pointfoot.flat.PF.pointfoot_flat_config import PointFootFlatCfg, PointFootFlatCfgPPO
#         task_registry.register("pointfoot_rough", PointFoot, PointFootRoughCfg(), PointFootRoughCfgPPO())
#         task_registry.register("pointfoot_flat", PointFoot, PointFootFlatCfg(), PointFootFlatCfgPPO())
#     elif robot_type.startswith("WF"):
#         from .pointfoot.WF.pointfoot import PointFoot
#         from legged_gym.envs.pointfoot.mixed_terrain.pointfoot_rough_config import PointFootRoughCfg, PointFootRoughCfgPPO
#         from legged_gym.envs.pointfoot.flat.WF.pointfoot_flat_config import PointFootFlatCfg, PointFootFlatCfgPPO
#         task_registry.register("pointfoot_flat", PointFoot, PointFootFlatCfg(), PointFootFlatCfgPPO())
#     elif robot_type.startswith("SF"):
#         from .pointfoot.SF.pointfoot import PointFoot
#         from legged_gym.envs.pointfoot.mixed_terrain.pointfoot_rough_config import PointFootRoughCfg, PointFootRoughCfgPPO
#         from legged_gym.envs.pointfoot.flat.SF.pointfoot_flat_config import PointFootFlatCfg, PointFootFlatCfgPPO
#         task_registry.register("pointfoot_flat", PointFoot, PointFootFlatCfg(), PointFootFlatCfgPPO())
#     else:
#         print("Error: Unknown robot type", robot_type)
#         sys.exit(1)
