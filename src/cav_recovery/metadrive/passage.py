"""A native MetaDrive road block: two directions share one physical strip.

The west bay is drivable in both directions during the clearance experiment.
Road edges are admissibility boundaries, checked at every Bullet substep; they
are not claimed to be impenetrable walls. No vehicle pose is edited after reset.
"""

from metadrive.component.block.base_block import BaseBlock
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.map.pg_map import PGMap
from metadrive.component.pgblock.pg_block import PGBlock
from metadrive.component.road_network import Road, NodeRoadNetwork
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.constants import PGLineType
from metadrive.component.pg_space import ParameterSpace


PASSAGE_LENGTH_M = 70.
PASSAGE_WIDTH_M = 3.4
BAY_END_X_M = 22.
BAY_CENTER_Y_M = 3.


class PassageBlock(BaseBlock):
    ID = "H"
    PARAMETER_SPACE = ParameterSpace({})
    pre_block_socket_index = 0

    @property
    def block_network_type(self):
        return NodeRoadNetwork

    def _sample_topology(self):
        for start, end, a, b in (
            ("west", "east", (0, 0), (PASSAGE_LENGTH_M, 0)),
            ("east", "west", (PASSAGE_LENGTH_M, 0), (0, 0)),
            ("bay_west", "bay_east", (0, BAY_CENTER_Y_M), (BAY_END_X_M, BAY_CENTER_Y_M)),
        ):
            self.block_network.add_lane(start, end, StraightLane(a, b, width=PASSAGE_WIDTH_M,
                                        line_types=(PGLineType.NONE, PGLineType.NONE)))
        self._respawn_roads = [Road("west", "east"), Road("east", "west")]
        return True

    create_in_world = PGBlock.create_in_world
    _construct_lane_line_in_block = PGBlock._construct_lane_line_in_block


class PassageMap(PGMap):
    def _generate(self):
        block = PassageBlock(0, self.road_network, random_seed=1)
        block.construct_block(self.engine.worldNP, self.engine.physics_world)
        self.blocks.append(block)


class PassageMapManager(PGMapManager):
    def reset(self):
        road_map = self.spawn_object(PassageMap, map_config=self.engine.global_config["map_config"], random_seed=None)
        self.load_map(road_map)
        road_map.spawn_roads = self.engine.global_config["spawn_roads"]


class SharedPassageEnv(MultiAgentMetaDrive):
    def setup_engine(self):
        super().setup_engine()
        self.engine.update_manager("map_manager", PassageMapManager())
